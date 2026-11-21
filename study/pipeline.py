"""Agentic evaluation pipeline: Generator agent -> Linguistic Evaluator agent.

All models are open-weight and run locally through llama.cpp (llama-cpp-python).
Prompt assembly and output validation are expressed as LangChain (LCEL) runnables.
Every call is appended to a JSONL log so runs are resumable and fully auditable.
"""
import base64, json, os, time, re, gc
from operator import itemgetter
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser
from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import MTMDChatHandler

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("MODELS_DIR", os.path.join(HERE, "models"))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
DATA = json.load(open(os.path.join(HERE, "golden_dataset_v2.json")))

MODEL_REGISTRY = {
    # name: (gguf, mmproj or None, licence, role)
    "qwen3vl-4b": ("Qwen3VL-4B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf", "Apache-2.0", "evaluator J1"),
    "phi4-mini": ("microsoft_Phi-4-mini-instruct-Q4_K_M.gguf", None, "MIT", "evaluator J2 (text only)"),
    "qwen3vl-2b": ("Qwen3VL-2B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", "Apache-2.0", "generator G1"),
    "smolvlm2-2.2b": ("SmolVLM2-2.2B-Instruct-Q4_K_M.gguf", "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf", "Apache-2.0", "generator G2"),
}
SEED = 20261109

def load_model(name, vision):
    gguf, mmproj, _, _ = MODEL_REGISTRY[name]
    kw = dict(n_ctx=4096, n_threads=2, n_batch=512, verbose=False, seed=SEED)
    if vision:
        kw["chat_handler"] = MTMDChatHandler(clip_model_path=os.path.join(MODELS, mmproj), verbose=False)
    return Llama(os.path.join(MODELS, gguf), **kw)

def image_uri(fname):
    return "data:image/png;base64," + base64.b64encode(open(os.path.join(HERE, "images", fname), "rb").read()).decode()

# ------------------------------------------------------------------ output schemas (constrained decoding)
class CostarScores(BaseModel):
    diagnosis: str = Field(max_length=200)
    C: int = Field(ge=1, le=5); O: int = Field(ge=1, le=5); S: int = Field(ge=1, le=5)
    T: int = Field(ge=1, le=5); A: int = Field(ge=1, le=5); R: int = Field(ge=1, le=5)

class Holistic(BaseModel):
    score: int = Field(ge=1, le=10)

class Pairwise(BaseModel):
    better: int = Field(ge=1, le=2)

# Compact GBNF grammars: force valid JSON with no optional whitespace (fewer decode steps on CPU).
DIG5 = '[1-5]'
G_COSTAR = r'''root ::= "{\"diagnosis\":\"" diag "\",\"C\":" d ",\"O\":" d ",\"S\":" d ",\"T\":" d ",\"A\":" d ",\"R\":" d "}"
diag ::= [^"\\\n]{8,140}
d ::= [1-5]
'''
G_HOL = r'''root ::= "{\"score\":" s "}"
s ::= [1-9] | "10"
'''
G_PAIR = r'''root ::= "{\"better\":" [12] "}"
'''

# ------------------------------------------------------------------ prompt templates (CO-STAR evaluator)
COSTAR_SYS = """You are the Linguistic Evaluator in a research pipeline that assesses the pragmatic competence of AI responses. You receive the user's PROMPT, a SPECIFICATION in CO-STAR format describing the situation and what an appropriate response must achieve, and the RESPONSE to evaluate.{mm_note}

Score the RESPONSE on six independent dimensions, each from 1 to 5:
C  Context fit: is the response coherent with, and grounded in, the situation{mm_c}?
O  Objective (illocutionary uptake): does it recognise what the user actually means, including any idiom, indirect request, implicature or irony, and accomplish the intended communicative act with correct content?
S  Style (register): are vocabulary and grammar suited to the specified style and medium?
T  Tone (politeness): is the attitude appropriate and respectful of the addressee?
A  Audience design: are complexity and amount of information calibrated to the specified audience?
R  Response format: does it follow the required format and length?

Scale: 5 = fully appropriate, 4 = minor problem, 3 = noticeable problem, 2 = serious problem, 1 = fails completely.
Score each dimension separately: a response can have a perfect tone yet fail the objective, or be correct yet in the wrong register. A literal reading of figurative or indirect language, a wrong fact or a misread chart is a failure of O. Do not reward length.
First write a diagnosis of at most 20 words, then the scores. Return only JSON."""
MM_NOTE_KEY = " The PROMPT refers to the attached image. An ANSWER KEY derived from the image's source data is provided."
MM_NOTE_NOKEY = " The PROMPT refers to the attached image; verify every claim against the image yourself."

SPEC_BLOCK = """PROMPT:
{prompt}

SPECIFICATION (CO-STAR):
Context: {context}
Objective: {objective}
Style: {style}
Tone: {tone}
Audience: {audience}
Response format: {response_format}{key_block}"""

COSTAR_HUMAN = SPEC_BLOCK + "\n\nRESPONSE:\n{response}"

SPEC_HOL_SYS = """You are an evaluator of AI responses. You receive the user's PROMPT, a SPECIFICATION describing the situation and what an appropriate response must achieve, and the RESPONSE.{mm_note}
Rate the overall quality of the RESPONSE as a reply to the PROMPT in this situation, from 1 (very poor) to 10 (excellent). Return only JSON."""

PLAIN_SYS = """You are an evaluator of AI assistant responses.{mm_note}
Rate the overall quality of the RESPONSE to the PROMPT from 1 (very poor) to 10 (excellent). Return only JSON."""
PLAIN_HUMAN = "PROMPT:\n{prompt}\n\nRESPONSE:\n{response}"

PAIR_SYS = """You are the Linguistic Evaluator in a research pipeline. You receive the user's PROMPT, a SPECIFICATION in CO-STAR format, and two candidate responses. Decide which response is more appropriate for this situation, considering whether it grasps what the user actually means, has correct content, and uses a suitable register, tone and format. Return only JSON with "better": 1 or 2."""
PAIR_HUMAN = SPEC_BLOCK + "\n\nRESPONSE 1:\n{r1}\n\nRESPONSE 2:\n{r2}"

GEN_SYS = "You are a helpful assistant."

def spec_vars(item, with_key=True):
    v = dict(prompt=item["prompt"], **item["costar"])
    v["key_block"] = f"\nANSWER KEY: {item['answer_key']}" if (item.get("answer_key") and with_key) else ""
    return v

# ------------------------------------------------------------------ LCEL plumbing
_ROLE = {"system": "system", "human": "user", "ai": "assistant"}

def make_invoker(llm, grammar_text, max_tokens):
    grammar = LlamaGrammar.from_string(grammar_text, verbose=False) if grammar_text else None
    def invoke(x):
        msgs = [{"role": _ROLE[m.type], "content": m.content} for m in x["pv"].to_messages()]
        if x.get("image"):
            last = msgs[-1]
            last["content"] = [{"type": "image_url", "image_url": {"url": x["image"]}}, {"type": "text", "text": last["content"]}]
        t = time.time()
        r = llm.create_chat_completion(messages=msgs, temperature=0.0, max_tokens=max_tokens, grammar=grammar, seed=SEED)
        x["_meta"] = dict(latency=round(time.time() - t, 2), usage=r["usage"], raw=r["choices"][0]["message"]["content"])
        return r["choices"][0]["message"]["content"]
    return invoke

def build_chain(llm, system, human, schema, grammar_text, max_tokens):
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    parser = PydanticOutputParser(pydantic_object=schema) if schema else None
    meta = {}
    def run(vars_, image=None):
        x = {"pv": prompt.invoke(vars_), "image": image}
        out = make_invoker(llm, grammar_text, max_tokens)(x)
        parsed = parser.parse(out).model_dump() if parser else out
        return parsed, x["_meta"]
    return run

class Log:
    def __init__(self, fname):
        self.path = os.path.join(RES, fname); self.done = set()
        if os.path.exists(self.path):
            for l in open(self.path):
                self.done.add(json.loads(l)["key"])
    def has(self, key): return key in self.done
    def add(self, rec):
        with open(self.path, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.done.add(rec["key"])
