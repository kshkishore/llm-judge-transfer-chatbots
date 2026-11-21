"""Figures for the JLM article: vector PDFs, Charis SIL Compact, readable in greyscale."""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FONTDIR = os.environ.get("CHARIS_DIR", "fonts")  # folder with the Charis SIL Compact TTF files
for f in ["CharisSILCompact-R.ttf", "CharisSILCompact-I.ttf", "CharisSILCompact-B.ttf", "CharisSILCompact-BI.ttf"]:
    fm.fontManager.addfont(os.path.join(FONTDIR, f))
name = fm.FontProperties(fname=os.path.join(FONTDIR, "CharisSILCompact-R.ttf")).get_name()
plt.rcParams.update({"font.family": name, "font.size": 7.5, "pdf.fonttype": 42, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": "#333333",
                     "mathtext.fontset": "custom", "mathtext.rm": name, "mathtext.it": name + ":italic"})
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
S = json.load(open(f"{RES}/summary_v2.json")); E1, E2 = S["exp1"], S["exp2"]
MM = 1 / 25.4; W = 110 * MM

def clean(ax, grid="x"):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color="#d0d0d0", lw=0.5, zorder=0); ax.set_axisbelow(True)

# ---- Figure 1: accuracy of measures on A-B and A-C contrasts
rows = [("Overlap and edit distance", None), ("BLEU", "bleu"), ("chrF", "chrf"), ("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"), ("ROUGE-L", "rougeL"),
        ("METEOR", "meteor"), ("TER", "ter"), ("Embedding and generative", None), ("BERTScore", "bertscore"), ("Sentence-BERT", "sbert"),
        ("BARTScore", "bartscore"), ("Natural language inference", None), ("Entailment", "nli_ent"), ("Bidirectional", "nli_bi"),
        ("Entailment − contradiction", "nli_e_minus_c"), ("Reference-free", None), ("GPT-2 perplexity", "gpt2_ppl"), ("Flesch Reading Ease", "fre"),
        ("Length", "n_words"), ("$f_{\\mathit{syntax}}$", "f_syntax"), ("LLM judge J1", None), ("Plain holistic", "J1_plain"),
        ("Holistic with specification", "J1_spec_holistic"), ("CO-STAR rubric", "J1_costar")]
def get(k):
    d = E1[k]["acc"] if k in E1 else E2[k]["acc"]
    return d
fig, ax = plt.subplots(figsize=(W, 112 * MM))
y = 0; ticks, labels, heads = [], [], []
for lab, k in rows:
    if k is None:
        heads.append((y, lab)); y -= 1.0; continue
    a = get(k)
    for kind, off, mk, fc in [("AB", 0.16, "o", "black"), ("AC", -0.16, "s", "white")]:
        v = a[kind]; ax.plot([v["ci"][0], v["ci"][1]], [y + off] * 2, color="#555555", lw=0.7, zorder=2)
        ax.scatter([v["acc"]], [y + off], marker=mk, s=16, facecolor=fc, edgecolor="black", linewidth=0.7, zorder=3)
    ticks.append(y); labels.append(lab); y -= 1.0
ax.axvline(0.5, color="#777777", lw=0.7, ls=(0, (1, 2)), zorder=1)
ax.set_yticks(ticks, labels); ax.set_xlim(-0.02, 1.02); ax.set_ylim(y + 0.4, 0.6)
for yy, lab in heads: ax.text(-0.02, yy, lab, ha="left", va="center", fontsize=7.5, style="italic", transform=ax.get_yaxis_transform() if False else ax.transData)
ax.set_xlabel("Pairwise accuracy"); clean(ax)
ax.scatter([], [], marker="o", s=16, facecolor="black", edgecolor="black", label="A–B (pragmalinguistic or grounding)")
ax.scatter([], [], marker="s", s=16, facecolor="white", edgecolor="black", label="A–C (sociopragmatic)")
ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.2)
ax.tick_params(axis="y", length=0)
fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/pragfail-fig1-metrics.pdf"); plt.close(fig)

# ---- Figure 2: distributions of variant scores
V = pd.read_csv(f"{RES}/variant_table_v2.csv")
panels = [("bleu", "BLEU", 1), ("bertscore", "BERTScore F1", 1), ("nli_bi", "Bidirectional entailment (%)", 1), ("J1_costar", "J1 CO-STAR $f_{\\mathit{prag}}$", 1)]
fig, axes = plt.subplots(2, 2, figsize=(W, 84 * MM))
rng = np.random.default_rng(20261109)
for ax, (col, title, mult) in zip(axes.flat, panels):
    for i, c in enumerate("ABC"):
        v = V[V.cand == c][col].dropna().values * mult
        if col == "bertscore" and np.nanmax(v) <= 1.0: v = v * 100
        x = i + rng.uniform(-0.22, 0.22, len(v))
        ax.scatter(x, v, s=3.5, color=["#222222", "#777777", "#aaaaaa"][i], alpha=0.75, linewidth=0, zorder=2)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="black", lw=1.6, zorder=3)
    ax.set_xticks([0, 1, 2], ["A", "B", "C"]); ax.set_title(title, fontsize=7.5, pad=3); clean(ax, "y")
fig.tight_layout(pad=0.3, h_pad=0.8, w_pad=1.0); fig.savefig(f"{OUT}/pragfail-fig2-variants.pdf"); plt.close(fig)

# ---- Figure 3: dimension profiles
D = list("COSTAR"); DN = ["Context", "Objective", "Style", "Tone", "Audience", "Format"]
fig, axes = plt.subplots(1, 2, figsize=(W, 52 * MM), sharey=True)
for ax, (j, ttl, sub) in zip(axes, [("J1", "J1 (Qwen3-VL-4B), 480 variants", V), ("J2", "J2 (Phi-4-mini), 432 text variants", V[V.category != "MM"])]):
    for c, ls, mk, fc in [("A", "-", "o", "black"), ("B", (0, (4, 2)), "^", "white"), ("C", (0, (1, 1.5)), "s", "#888888")]:
        m = [sub[sub.cand == c][f"{j}_costar_{d}"].mean() for d in D]
        ax.plot(range(6), m, ls=ls, color="black", lw=0.9, marker=mk, ms=3.6, mfc=fc, mec="black", mew=0.7, label=c)
    ax.set_xticks(range(6), DN, rotation=35, ha="right"); ax.set_ylim(1, 5.15); ax.set_title(ttl, fontsize=7.5, pad=3); clean(ax, "y")
axes[0].set_ylabel("Mean score (1–5)")
axes[1].legend(title=None, frameon=False, fontsize=7, loc="lower right", handlelength=2.6)
fig.tight_layout(pad=0.3, w_pad=0.8); fig.savefig(f"{OUT}/pragfail-fig3-profiles.pdf"); plt.close(fig)
print("ok", name)
