"""All automatic metrics of the main study on the expanded Golden Dataset.
Usage: python metrics_v2.py contrast   -> results/metrics_contrast.json (480 variants)
       python metrics_v2.py gen        -> results/metrics_gen.json (320 generated responses)"""
import json, os, sys
from metrics import overlap, f_syntax, strip_markdown
import metrics_ext as mx
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
data = json.load(open(os.path.join(HERE, "golden_dataset_v2.json"))); byid = {it["id"]: it for it in data}
what = sys.argv[1]; rows, cands, refs, images = [], [], [], []
if what == "contrast":
    for it in data:
        for k in "ABC":
            rows.append(dict(item=it["id"], category=it["category"], origin=it["origin"], cand=k)); cands.append(it["candidates"][k]); refs.append(it["reference"]); images.append(it.get("image"))
    out = "metrics_contrast.json"
else:
    for f in ["gen_qwen3vl-2b.jsonl", "gen_smolvlm2-2.2b.jsonl"]:
        for l in open(os.path.join(RES, f)):
            g = json.loads(l); it = byid[g["item"]]
            rows.append(dict(item=g["item"], category=it["category"], origin=it["origin"], generator=g["generator"]))
            cands.append(strip_markdown(g["output"]) if g["output"].strip() else "(empty)"); refs.append(it["reference"]); images.append(it.get("image"))
    out = "metrics_gen.json"
print("base metrics", flush=True)
for row, o in zip(rows, overlap(cands, refs)): row.update(o)
for row, c in zip(rows, cands): row["f_syntax"] = f_syntax(c)
mx.run(rows, cands, refs, images, out)
