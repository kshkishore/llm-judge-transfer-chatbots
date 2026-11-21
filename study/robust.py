"""Experiment 6: robustness of the evaluator (text items only, 72 response variants per condition).

Conditions for J1 (Qwen3-VL-4B):
  costar_nodiag   scores only, no diagnosis before scoring
  costar_para     the rubric prompt reworded (same content, different wording)
  costar_rev      rubric dimensions and output keys in reverse order (R, A, T, S, O, C)
  costar_s1..s3   original rubric, sampled at temperature 0.7 with three different seeds
Conditions for J2 (Phi-4-mini): spec_holistic and plain (replication of Experiment 2 with a second judge).
Usage: python robust.py <j1|j2>
"""
import sys, gc, time
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from llama_cpp import LlamaGrammar
from pipeline import *

TEXT_ITEMS = [it for it in DATA if it["category"] != "MM"]

class ScoresOnly(BaseModel):
    C: int = Field(ge=1, le=5); O: int = Field(ge=1, le=5); S: int = Field(ge=1, le=5)
    T: int = Field(ge=1, le=5); A: int = Field(ge=1, le=5); R: int = Field(ge=1, le=5)

G_NODIAG = r'''root ::= "{\"C\":" d ",\"O\":" d ",\"S\":" d ",\"T\":" d ",\"A\":" d ",\"R\":" d "}"
d ::= [1-5]
'''
G_REV = r'''root ::= "{\"diagnosis\":\"" diag "\",\"R\":" d ",\"A\":" d ",\"T\":" d ",\"S\":" d ",\"O\":" d ",\"C\":" d "}"
diag ::= [^"\\\n]{8,140}
d ::= [1-5]
'''

NODIAG_SYS = COSTAR_SYS.replace("First write a diagnosis of at most 20 words, then the scores. Return only JSON.", "Return only JSON with the six scores.")

PARA_SYS = """Your task is to judge how appropriately an AI system has answered a user. Three things are given: the PROMPT the user wrote; a CO-STAR SPECIFICATION that describes the situation and states what a suitable answer has to accomplish; and the RESPONSE that you must assess.

Give six separate ratings on a scale of 1 to 5 (5 = entirely suitable, 4 = slight flaw, 3 = clear flaw, 2 = major flaw, 1 = complete failure):
C  Context fit - does the answer make sense in this situation and stay anchored to it?
O  Objective (illocutionary uptake) - has the answer understood what the user really intends, including idioms, indirect requests, implied meanings and irony, and does it carry out the required communicative act with accurate content?
S  Style (register) - do the choice of words and the grammar match the style and medium that were specified?
T  Tone (politeness) - is the attitude fitting, and does it show respect for the person addressed?
A  Audience design - are the difficulty and the quantity of information right for the stated audience?
R  Response format - are the required format and length observed?

Treat the six ratings as independent of one another: an answer may be flawless in tone and still miss the objective, or be accurate and still use the wrong register. Taking figurative or indirect language literally, stating a wrong fact or misreading a chart counts as failing O. Longer answers do not deserve higher ratings.
Begin with a diagnosis of no more than 20 words and then give the ratings. Output JSON only."""

_lines = COSTAR_SYS.format(mm_note="", mm_c="").split("\n")
_dims = [l for l in _lines if l[:3] in ("C  ", "O  ", "S  ", "T  ", "A  ", "R  ")]
REV_SYS = "\n".join(_lines).replace("\n".join(_dims), "\n".join(_dims[::-1]))
assert REV_SYS.index("R  Response format") < REV_SYS.index("C  Context fit")

def esc(s): return s.replace("{", "{{").replace("}", "}}")

def chain(llm, system, human, schema, grammar_text, max_tokens, temperature=0.0, seed=SEED):
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    parser = PydanticOutputParser(pydantic_object=schema)
    grammar = LlamaGrammar.from_string(grammar_text, verbose=False)
    def run(vars_):
        msgs = [{"role": {"system": "system", "human": "user"}[m.type], "content": m.content} for m in prompt.invoke(vars_).to_messages()]
        t = time.time()
        r = llm.create_chat_completion(messages=msgs, temperature=temperature, max_tokens=max_tokens, grammar=grammar, seed=seed)
        raw = r["choices"][0]["message"]["content"]
        return parser.parse(raw).model_dump(), dict(latency=round(time.time() - t, 2), usage=r["usage"], raw=raw)
    return run

def run_conditions(judge, conds, log_name):
    log = Log(log_name); llm = load_model(judge, False)
    base = COSTAR_SYS.format(mm_note="", mm_c="")
    for cond in conds:
        if cond == "costar_nodiag": ch = chain(llm, esc(NODIAG_SYS.format(mm_note="", mm_c="")), COSTAR_HUMAN, ScoresOnly, G_NODIAG, 40)
        elif cond == "costar_para": ch = chain(llm, esc(PARA_SYS), COSTAR_HUMAN, CostarScores, G_COSTAR, 110)
        elif cond == "costar_rev":  ch = chain(llm, esc(REV_SYS), COSTAR_HUMAN, CostarScores, G_REV, 110)
        elif cond.startswith("costar_s"):
            ch = chain(llm, esc(base), COSTAR_HUMAN, CostarScores, G_COSTAR, 110, temperature=0.7, seed={"costar_s1": 101, "costar_s2": 202, "costar_s3": 303}[cond])
        elif cond == "spec_holistic": ch = chain(llm, esc(SPEC_HOL_SYS.format(mm_note="")), COSTAR_HUMAN, Holistic, G_HOL, 8)
        elif cond == "plain": ch = chain(llm, esc(PLAIN_SYS.format(mm_note="")), PLAIN_HUMAN, Holistic, G_HOL, 8)
        for it in TEXT_ITEMS:
            for cand in "ABC":
                key = f"{judge}|{cond}|{it['id']}|{cand}"
                if log.has(key): continue
                v = spec_vars(it); v["response"] = it["candidates"][cand]
                parsed, meta = ch(v)
                log.add(dict(key=key, judge=judge, condition=cond, item=it["id"], category=it["category"], cand=cand, scores=parsed, **meta))
                print(key, {k: v for k, v in parsed.items() if k != "diagnosis"}, meta["latency"], flush=True)
    del llm; gc.collect()

if __name__ == "__main__":
    if sys.argv[1] == "j1": run_conditions("qwen3vl-4b", ["costar_nodiag", "costar_para", "costar_rev", "costar_s1", "costar_s2", "costar_s3"], "robust_j1.jsonl")
    if sys.argv[1] == "j2": run_conditions("phi4-mini", ["spec_holistic", "plain"], "robust_j2.jsonl")
    if sys.argv[1] == "check":
        print(REV_SYS); print("-----"); print(NODIAG_SYS.format(mm_note="", mm_c="")[-200:])
        import inspect
        from llama_cpp import Llama
        sig = inspect.signature(Llama.create_chat_completion)
        print({k: sig.parameters[k].default for k in ["temperature", "top_p", "top_k", "min_p", "typical_p", "repeat_penalty", "presence_penalty", "frequency_penalty"]})
