"""Builds the expanded Golden Dataset (golden_dataset_v2.json).

The 32 pilot items (golden_dataset.json) and the 16 items of the pilot's held-out set (golden_dataset_ext.json)
are carried over verbatim; the new items are defined in items_*.py. New chart items are rendered from the data
stored in each item, so the answer key and the image come from the same numbers. Every item passes the same
automated checks, and the variant design rules that can be measured are verified item by item.
"""
import json, os, glob, importlib.util, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sacrebleu.metrics import CHRF

HERE = os.path.dirname(os.path.abspath(__file__)); IMG = os.path.join(HERE, "images")
FIELDS = ["context", "objective", "style", "tone", "audience", "response_format"]

def T(id, phen, layer, illoc, prompt, costar, ref, A, B, C, image=None, key=None, chart=None):
    return dict(id=id, category=id.split("-")[0], phenomenon=phen, pragmatic_layer=layer,
                expected_illocution=illoc, prompt=prompt, image=image, answer_key=key, chart=chart,
                costar=dict(zip(FIELDS, costar)), reference=ref, candidates={"A": A, "B": B, "C": C})

def render(item):
    c = item["chart"]; fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=110); col = "#2F5597"
    if c["kind"] == "bar":
        b = ax.bar(c["x"], c["y"], color=col); ax.bar_label(b, fontsize=9)
    elif c["kind"] == "barh":
        b = ax.barh(c["x"][::-1], c["y"][::-1], color=col); ax.bar_label(b, fontsize=9); ax.set_xlabel(c.get("ylabel", ""))
    elif c["kind"] == "line":
        ax.plot(c["x"], c["y"], marker="o", color=col)
        for xi, yi in zip(c["x"], c["y"]): ax.annotate(str(yi), (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    elif c["kind"] == "pie":
        ax.pie(c["y"], labels=[f"{l} ({v}%)" for l, v in zip(c["x"], c["y"])], startangle=90, counterclock=False,
               colors=["#2F5597", "#5B9BD5", "#A5A5A5", "#ED7D31", "#FFC000", "#70AD47"][:len(c["y"])], textprops={"fontsize": 9}); ax.axis("equal")
    elif c["kind"] == "grouped":
        xs = np.arange(len(c["x"])); n = len(c["series"]); w = 0.8 / n
        for i, (name, vals) in enumerate(c["series"].items()):
            b = ax.bar(xs + (i - (n - 1) / 2) * w, vals, w, label=name, color=["#2F5597", "#ED7D31", "#A5A5A5"][i]); ax.bar_label(b, fmt=c.get("fmt", "%d"), fontsize=9)
        ax.set_xticks(xs, c["x"]); ax.legend(frameon=False); ax.set_ylim(0, c.get("ymax", max(max(v) for v in c["series"].values()) * 1.25))
    elif c["kind"] == "scatter":
        ax.scatter(c["x"], c["y"], color=col, s=28); ax.set_xlabel(c["xlabel"])
    if c["kind"] not in ("pie", "barh"): ax.set_ylabel(c.get("ylabel", ""))
    if c["kind"] == "scatter" and c.get("ylim"): ax.set_ylim(*c["ylim"])
    ax.set_title(c["title"], fontsize=11)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.text(0.99, 0.01, "Illustrative data", ha="right", fontsize=7, color="#666666")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, item["image"])); plt.close(fig)

if __name__ == "__main__":
    pilot = json.load(open(os.path.join(HERE, "golden_dataset.json"))); held = json.load(open(os.path.join(HERE, "golden_dataset_ext.json")))
    for it in pilot: it["origin"] = "pilot"
    for it in held: it["origin"] = "pilot-heldout"
    new = []
    for f in sorted(glob.glob(os.path.join(HERE, "items_*.py"))):
        spec = importlib.util.spec_from_file_location(os.path.basename(f)[:-3], f); m = importlib.util.module_from_spec(spec)
        m.T = T; sys.modules[spec.name] = m; spec.loader.exec_module(m)
        for it in m.NEW: it["origin"] = "new"; new.append(it)
    D = pilot + held + new
    for it in new:
        if it["image"]: render(it)
    REQ = {"id", "category", "phenomenon", "pragmatic_layer", "expected_illocution", "prompt", "costar", "reference", "candidates"}
    chrf = CHRF(); ids = set(); prompts = set(); warn = []
    for it in D:
        assert REQ <= it.keys(), it["id"]
        assert it["id"] not in ids, it["id"]; ids.add(it["id"])
        assert it["prompt"] not in prompts, it["id"]; prompts.add(it["prompt"])
        assert len(it["costar"]) == 6 and all(it["costar"][k].strip() for k in FIELDS), it["id"]
        assert set(it["candidates"]) == {"A", "B", "C"} and all(v.strip() for v in it["candidates"].values()), it["id"]
        if it["category"] == "MM": assert it["answer_key"] and os.path.exists(os.path.join(IMG, it["image"])), it["id"]
        if it["origin"] != "new": continue
        for s in [it["prompt"], it["reference"], *it["candidates"].values(), *it["costar"].values()]:
            assert s == s.strip() and "  " not in s, (it["id"], s)
            assert not any(ch in s for ch in "\u00a0\u2018\u2019\u201c\u201d\u2013\u2014"), (it["id"], "typographic character", s)
        A, B, C, R = it["candidates"]["A"], it["candidates"]["B"], it["candidates"]["C"], it["reference"]
        # Rule 1/2: B is closer to the reference than A; A is not a near-copy of the reference
        ca, cb = chrf.sentence_score(A, [R]).score, chrf.sentence_score(B, [R]).score
        assert cb > ca + 10, (it["id"], "B not closer to reference than A", round(ca, 1), round(cb, 1))
        assert ca < 60, (it["id"], "A too close to reference", round(ca, 1))
        # Rule 4: A and B of similar length
        la, lb = len(A.split()), len(B.split())
        if abs(la - lb) > max(6, 0.4 * max(la, lb)): warn.append((it["id"], "A/B length", la, lb))
    json.dump(D, open(os.path.join(HERE, "golden_dataset_v2.json"), "w"), indent=1, ensure_ascii=False)
    from collections import Counter
    print(len(D), "items;", Counter(i["category"] for i in D)); print("origin:", Counter(i["origin"] for i in D))
    if warn: print("warnings:", warn)
