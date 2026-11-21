"""Re-runs a random sample of the pilot calls carried into the main study and compares each fresh output with the logged
one character for character. Writes results/reuse_check.json."""
import json, os, random, gc, time
from pipeline import *
from run import TEXT, MMF
byid = {it["id"]: it for it in DATA}
def jl(f): return [json.loads(l) for l in open(os.path.join(RES, f))]
rng = random.Random(7)
c1 = [r for r in jl("contrast_j1.jsonl") if r.get("reused_from_pilot")]
c2 = [r for r in jl("contrast_j2.jsonl") if r.get("reused_from_pilot")]
sample = rng.sample([r for r in c1 if r["category"] != "MM" and r["condition"] == "costar"], 6) + \
         rng.sample([r for r in c1 if r["category"] != "MM" and r["condition"] in ("plain", "spec_holistic")], 2) + \
         rng.sample([r for r in c1 if r["category"] == "MM" and r["condition"] == "costar"], 2)
sample2 = rng.sample([r for r in c2 if r["condition"] == "costar"], 2) + rng.sample([r for r in c2 if r["condition"] != "costar"], 1)
out = []
def check(judge, vision, rows):
    llm = load_model(judge, vision)
    for r in rows:
        it = byid[r["item"]]; mm = it["category"] == "MM"; cond = r["condition"]
        note = MM_NOTE_KEY if mm else ""
        if cond == "costar":
            sysm = COSTAR_SYS.format(mm_note=note, mm_c=" and, if an image is given, are all claims about it correct" if mm else "")
            ch = build_chain(llm, sysm.replace("{", "{{").replace("}", "}}"), COSTAR_HUMAN, CostarScores, G_COSTAR, 110)
        elif cond == "spec_holistic":
            ch = build_chain(llm, SPEC_HOL_SYS.format(mm_note=note).replace("{", "{{").replace("}", "}}"), COSTAR_HUMAN, Holistic, G_HOL, 8)
        else:
            ch = build_chain(llm, PLAIN_SYS.format(mm_note=" The PROMPT refers to the attached image." if mm else ""), PLAIN_HUMAN, Holistic, G_HOL, 8)
        v = spec_vars(it, True); v["response"] = it["candidates"][r["cand"]]
        parsed, meta = ch(v, image_uri(it["image"]) if mm else None)
        out.append(dict(key=r["key"], identical=meta["raw"] == r["raw"], logged=r["raw"], fresh=meta["raw"]))
        print(r["key"], meta["raw"] == r["raw"], flush=True)
    del llm; gc.collect()
check("qwen3vl-4b", False, [r for r in sample if r["category"] != "MM"])
check("qwen3vl-4b", True, [r for r in sample if r["category"] == "MM"])
check("phi4-mini", False, sample2)
# generators
g = rng.choice([r for r in jl("gen_qwen3vl-2b.jsonl") if r["category"] != "MM"])
llm = load_model("qwen3vl-2b", True); it = byid[g["item"]]
rr = llm.create_chat_completion(messages=[{"role": "system", "content": GEN_SYS}, {"role": "user", "content": it["prompt"]}], temperature=0.0, max_tokens=180, seed=SEED)
txt = rr["choices"][0]["message"]["content"].strip(); out.append(dict(key=g["key"], identical=txt == g["output"], logged=g["output"], fresh=txt)); print(g["key"], txt == g["output"])
json.dump(dict(n=len(out), identical=sum(o["identical"] for o in out), calls=out), open(os.path.join(RES, "reuse_check.json"), "w"), indent=1)
print("identical", sum(o["identical"] for o in out), "of", len(out))
