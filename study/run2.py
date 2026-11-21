"""Main study runner: every phase over the expanded Golden Dataset (golden_dataset_v2.json).

Phases are resumable: a call whose key is already in its log is skipped. Logs of the pilot study were carried over
for the 48 items taken over verbatim, because the configuration is deterministic (Section 3.6); a sample of reused
calls is re-run in check_reuse.py to confirm that each reproduces its logged output exactly.
Usage: python run2.py <phase>
"""
import sys, json, os, random
from run import judge_contrast, generate, judge_generations, pairwise, TEXT, MMF
from robust import run_conditions
import robust
from pipeline import DATA, HERE

def rob_subset():
    """Stratified robustness subset: four text items per text category, drawn with a fixed seed."""
    rng = random.Random(20261109); out = []
    cats = sorted({it["category"] for it in DATA if it["category"] != "MM"})
    for c in cats:
        ids = sorted(it["id"] for it in DATA if it["category"] == c)
        out += rng.sample(ids, 4)
    return out

if __name__ == "__main__":
    ph = sys.argv[1]
    if ph == "j1_text": judge_contrast("qwen3vl-4b", False, ["costar"], TEXT, "contrast_j1.jsonl")
    if ph == "j1_mm":   judge_contrast("qwen3vl-4b", True, ["costar", "costar_nokey", "spec_holistic", "plain"], MMF, "contrast_j1.jsonl")
    if ph == "j1_hol":  judge_contrast("qwen3vl-4b", False, ["spec_holistic", "plain"], TEXT, "contrast_j1.jsonl")
    if ph == "j2":      judge_contrast("phi4-mini", False, ["costar", "spec_holistic", "plain"], TEXT, "contrast_j2.jsonl")
    if ph == "gen":
        generate("qwen3vl-2b", lambda it: True, "gen_qwen3vl-2b.jsonl")
        generate("smolvlm2-2.2b", lambda it: True, "gen_smolvlm2-2.2b.jsonl")
    if ph == "j1_gen":
        judge_generations("qwen3vl-4b", False, TEXT, "genjudge_j1.jsonl")
        judge_generations("qwen3vl-4b", True, MMF, "genjudge_j1.jsonl")
    if ph == "j2_gen":  judge_generations("phi4-mini", False, TEXT, "genjudge_j2.jsonl")
    if ph == "pair":    pairwise("qwen3vl-4b", "pairwise_j1.jsonl")
    if ph == "rob":
        sub = set(rob_subset()); json.dump(sorted(sub), open(os.path.join(HERE, "results", "robust_subset.json"), "w"))
        robust.TEXT_ITEMS[:] = [it for it in DATA if it["id"] in sub]
        run_conditions("qwen3vl-4b", ["costar_para", "costar_s1", "costar_s2", "costar_s3"], "robust_j1.jsonl")
