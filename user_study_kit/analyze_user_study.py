"""Analysis of the user study (see PROTOCOL.md).

Usage:  python analyze_user_study.py responses.csv [key.csv]
responses.csv: long format with columns
    participant, stimulus_id, q1_understanding, q2_tone, q3_form, q4_overall, q5_use
Writes user_study_results.json and prints a summary.
Requires: pandas, numpy, scipy, krippendorff
"""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu
import krippendorff

resp = pd.read_csv(sys.argv[1])
key = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "key.csv")
Q = ["q1_understanding", "q2_tone", "q3_form", "q4_overall"]
out = {"n_participants": int(resp.participant.nunique()), "n_ratings": int(len(resp))}

# 1. Inter-rater reliability (ordinal alpha per question)
out["alpha"] = {}
for q in Q:
    m = resp.pivot_table(index="participant", columns="stimulus_id", values=q).to_numpy(dtype=float)
    out["alpha"][q] = round(float(krippendorff.alpha(reliability_data=m, level_of_measurement="ordinal")), 3)
if "q5_use" in resp:
    u = resp.assign(q5=resp.q5_use.astype(str).str.lower().isin(["1", "yes", "true"]).astype(float))
    m = u.pivot_table(index="participant", columns="stimulus_id", values="q5").to_numpy(dtype=float)
    out["alpha"]["q5_use"] = round(float(krippendorff.alpha(reliability_data=m, level_of_measurement="nominal")), 3)

# per-reply means
agg = resp.groupby("stimulus_id")[Q].mean()
if "q5_use" in resp:
    agg["q5_use_rate"] = resp.assign(q5=resp.q5_use.astype(str).str.lower().isin(["1", "yes", "true"])).groupby("stimulus_id").q5.mean()
d = key.merge(agg, left_on="stimulus_id", right_index=True)

rng = np.random.default_rng(1)
def boot_rho(x, y, n=5000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    r = spearmanr(x, y)[0]; bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x)); bs.append(spearmanr(x[i], y[i])[0])
    return round(float(r), 3), [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)], int(len(x))

# 2. U1: agreement with judges (J2 has no chart scores -> compare on text items)
txt = d[d.J2.notna()]
out["U1"] = {
    "J1_vs_q4_all": boot_rho(d.J1, d.q4_overall),
    "J1_vs_q4_text": boot_rho(txt.J1, txt.q4_overall),
    "J2_vs_q4_text": boot_rho(txt.J2, txt.q4_overall),
    "J1_O_vs_q1_text": boot_rho(txt.J1_O, txt.q1_understanding),
    "J2_O_vs_q1_text": boot_rho(txt.J2_O, txt.q1_understanding),
}
diffs = []
x1, x2, y = txt.J1.to_numpy(float), txt.J2.to_numpy(float), txt.q4_overall.to_numpy(float)
for _ in range(5000):
    i = rng.integers(0, len(y), len(y))
    diffs.append(spearmanr(x1[i], y[i])[0] - spearmanr(x2[i], y[i])[0])
out["U1"]["rho_J1_minus_J2_CI"] = [round(float(np.nanpercentile(diffs, 2.5)), 3), round(float(np.nanpercentile(diffs, 97.5)), 3)]

# 3. U2: chatbots compared by users (paired by item)
p = d.pivot_table(index="item", columns="chatbot", values="q4_overall")
w = wilcoxon(p.G1, p.G2)
out["U2"] = {"mean_G1": round(float(p.G1.mean()), 3), "mean_G2": round(float(p.G2.mean()), 3),
             "G1_higher": int((p.G1 > p.G2).sum()), "G2_higher": int((p.G2 > p.G1).sum()), "p": float(w.pvalue)}

# 4. U3: format rule vs perceived form
ok = d.format_rule_ok.astype(str).str.lower().isin(["true", "1"])
a, b = d.loc[ok, "q3_form"], d.loc[~ok, "q3_form"]
mw = mannwhitneyu(a, b)
out["U3"] = {"n_pass": int(ok.sum()), "n_fail": int((~ok).sum()), "mean_pass": round(float(a.mean()), 3),
             "mean_fail": round(float(b.mean()), 3), "auc": round(float(mw.statistic / (len(a) * len(b))), 3), "p": float(mw.pvalue)}

# 5. U4: misreading flags (Objective <= 2)
def grp(r):
    j1 = r.J1_O <= 2; j2 = (r.J2_O <= 2) if pd.notna(r.J2_O) else False
    return "both" if j1 and j2 else "J1_only" if j1 else "J2_only" if j2 else "neither"
d["flag"] = d.apply(grp, axis=1)
out["U4"] = {g: {"n": int((d.flag == g).sum()), "mean_q1": round(float(d.loc[d.flag == g, "q1_understanding"].mean()), 3)}
             for g in ["both", "J1_only", "J2_only", "neither"] if (d.flag == g).any()}

# 6. metrics
out["metrics_vs_q4"] = {m: boot_rho(d[m], d.q4_overall, 2000) for m in ["bleu", "bertscore", "nli_bi", "n_words"]}

json.dump(out, open("user_study_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
