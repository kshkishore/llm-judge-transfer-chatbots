"""Carries the pilot's logged calls for the 48 reused items into the main-study logs (same keys, same configuration).
Only conditions that the main study also runs are carried over. Run once, before run2.py."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "development_logs"); R = os.path.join(HERE, "results")
def jl(f): return [json.loads(l) for l in open(os.path.join(P, f))]
def write(name, rows):
    path = os.path.join(R, name); have = set()
    if os.path.exists(path): have = {json.loads(l)["key"] for l in open(path)}
    n = 0
    with open(path, "a") as f:
        for r in rows:
            if r["key"] in have: continue
            r = dict(r, reused_from_pilot=True); f.write(json.dumps(r, ensure_ascii=False) + "\n"); have.add(r["key"]); n += 1
    print(name, "+", n)
write("contrast_j1.jsonl", jl("contrast_j1.jsonl") + jl("extset_j1.jsonl"))
write("contrast_j2.jsonl", jl("contrast_j2.jsonl") + [r for r in jl("robust_j2.jsonl") if r["condition"] in ("spec_holistic", "plain")] + jl("extset_j2.jsonl"))
for f in ["gen_qwen3vl-2b.jsonl", "gen_smolvlm2-2.2b.jsonl", "genjudge_j1.jsonl", "genjudge_j2.jsonl", "pairwise_j1.jsonl"]: write(f, jl(f))
write("robust_j1.jsonl", [r for r in jl("robust_j1.jsonl") if r["condition"] in ("costar_para", "costar_s1", "costar_s2", "costar_s3")])
