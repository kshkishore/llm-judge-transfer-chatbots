"""Runs one phase of the pilot. Usage: python run.py <phase>
Phases are resumable: completed calls are skipped using the JSONL logs in results/.
"""
import sys, json, os, time, gc
from pipeline import *

def items(cat_filter):
    return [it for it in DATA if cat_filter(it)]
TEXT = lambda it: it["category"] != "MM"
MMF = lambda it: it["category"] == "MM"

def judge_contrast(judge, vision, conditions, sel, log_name):
    log = Log(log_name); llm = load_model(judge, vision); n = 0
    for cond in conditions:
        for it in items(sel):
            mm = it["category"] == "MM"
            img = image_uri(it["image"]) if mm else None
            with_key = cond != "costar_nokey"
            note = (MM_NOTE_KEY if with_key else MM_NOTE_NOKEY) if mm else ""
            if cond.startswith("costar"):
                sysm = COSTAR_SYS.format(mm_note=note, mm_c=" and, if an image is given, are all claims about it correct" if mm else "")
                ch = build_chain(llm, sysm.replace("{", "{{").replace("}", "}}"), COSTAR_HUMAN, CostarScores, G_COSTAR, 110)
            elif cond == "spec_holistic":
                ch = build_chain(llm, SPEC_HOL_SYS.format(mm_note=note).replace("{", "{{").replace("}", "}}"), COSTAR_HUMAN, Holistic, G_HOL, 8)
            elif cond == "plain":
                ch = build_chain(llm, PLAIN_SYS.format(mm_note=" The PROMPT refers to the attached image." if mm else ""), PLAIN_HUMAN, Holistic, G_HOL, 8)
            for cand in "ABC":
                key = f"{judge}|{cond}|{it['id']}|{cand}"
                if log.has(key): continue
                v = spec_vars(it, with_key); v["response"] = it["candidates"][cand]
                parsed, meta = ch(v, img)
                log.add(dict(key=key, judge=judge, condition=cond, item=it["id"], category=it["category"], cand=cand, scores=parsed, **meta))
                n += 1; print(key, parsed, meta["latency"], flush=True)
    del llm; gc.collect(); return n

def generate(gen, sel, log_name):
    log = Log(log_name); llm = load_model(gen, True)
    for it in items(sel):
        key = f"{gen}|{it['id']}"
        if log.has(key): continue
        content = it["prompt"]
        # SmolVLM2's chat template renders only list-structured content; plain strings are silently dropped.
        structured = gen == "smolvlm2-2.2b"
        msgs = [{"role": "system", "content": [{"type": "text", "text": GEN_SYS}] if structured else GEN_SYS}]
        if it["image"]:
            msgs.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_uri(it["image"])}}, {"type": "text", "text": content}]})
        else:
            msgs.append({"role": "user", "content": [{"type": "text", "text": content}] if structured else content})
        t = time.time()
        r = llm.create_chat_completion(messages=msgs, temperature=0.0, max_tokens=180, seed=SEED)
        txt = r["choices"][0]["message"]["content"].strip()
        log.add(dict(key=key, generator=gen, item=it["id"], category=it["category"], output=txt, finish=r["choices"][0]["finish_reason"],
                     latency=round(time.time() - t, 2), usage=r["usage"]))
        print(key, round(time.time() - t, 1), repr(txt[:160]), flush=True)
    del llm; gc.collect()

def judge_generations(judge, vision, sel, log_name):
    log = Log(log_name); llm = load_model(judge, vision)
    gens = [json.loads(l) for f in ["gen_qwen3vl-2b.jsonl", "gen_smolvlm2-2.2b.jsonl"] for l in open(os.path.join(RES, f))]
    byid = {it["id"]: it for it in DATA}
    for g in gens:
        it = byid[g["item"]]
        if not sel(it): continue
        key = f"{judge}|costar|{g['generator']}|{it['id']}"
        if log.has(key): continue
        mm = it["category"] == "MM"
        sysm = COSTAR_SYS.format(mm_note=MM_NOTE_KEY if mm else "", mm_c=" and, if an image is given, are all claims about it correct" if mm else "")
        ch = build_chain(llm, sysm.replace("{", "{{").replace("}", "}}"), COSTAR_HUMAN, CostarScores, G_COSTAR, 110)
        v = spec_vars(it, True); v["response"] = g["output"][:1500]
        parsed, meta = ch(v, image_uri(it["image"]) if mm else None)
        log.add(dict(key=key, judge=judge, generator=g["generator"], item=it["id"], category=it["category"], scores=parsed, **meta))
        print(key, parsed, meta["latency"], flush=True)
    del llm; gc.collect()

def pairwise(judge, log_name):
    log = Log(log_name); llm = load_model(judge, False)
    ch = build_chain(llm, PAIR_SYS, PAIR_HUMAN, Pairwise, G_PAIR, 8)
    for it in items(TEXT):
        for other in "BC":
            for order in ("A-first", "A-second"):
                key = f"{judge}|pair|{it['id']}|A{other}|{order}"
                if log.has(key): continue
                v = spec_vars(it); a, o = it["candidates"]["A"], it["candidates"][other]
                v["r1"], v["r2"] = (a, o) if order == "A-first" else (o, a)
                parsed, meta = ch(v)
                a_wins = (parsed["better"] == 1) == (order == "A-first")
                log.add(dict(key=key, judge=judge, item=it["id"], category=it["category"], pair="A" + other, order=order, choice=parsed["better"], a_wins=a_wins, **meta))
                print(key, parsed, a_wins, meta["latency"], flush=True)
    del llm; gc.collect()

if __name__ == "__main__":
    ph = sys.argv[1]
    if ph == "j1_text":  judge_contrast("qwen3vl-4b", False, ["costar", "spec_holistic", "plain"], TEXT, "contrast_j1.jsonl")
    if ph == "j1_mm":    judge_contrast("qwen3vl-4b", True, ["costar", "costar_nokey", "spec_holistic", "plain"], MMF, "contrast_j1.jsonl")
    if ph == "gen":
        generate("qwen3vl-2b", lambda it: True, "gen_qwen3vl-2b.jsonl")
        generate("smolvlm2-2.2b", lambda it: True, "gen_smolvlm2-2.2b.jsonl")
    if ph == "j1_gen":
        judge_generations("qwen3vl-4b", False, TEXT, "genjudge_j1.jsonl")
        judge_generations("qwen3vl-4b", True, MMF, "genjudge_j1.jsonl")
    if ph == "j2":
        judge_contrast("phi4-mini", False, ["costar"], TEXT, "contrast_j2.jsonl")
        judge_generations("phi4-mini", False, TEXT, "genjudge_j2.jsonl")
    if ph == "pair":     pairwise("qwen3vl-4b", "pairwise_j1.jsonl")
    if ph == "smoke":
        import pipeline
        it = DATA[0]; llm = load_model("qwen3vl-4b", False)
        sysm = COSTAR_SYS.format(mm_note="", mm_c="")
        ch = build_chain(llm, sysm.replace("{", "{{").replace("}", "}}"), COSTAR_HUMAN, CostarScores, G_COSTAR, 110)
        for c in "ABC":
            v = spec_vars(it); v["response"] = it["candidates"][c]
            print(c, ch(v))
