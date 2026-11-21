"""Analysis of the main study (160 items). Writes results/summary_v2.json and per-variant/per-response tables.
Every number reported in Chapter 4 is read from the logs through this script."""
import json, os, re, random
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon, binomtest, mannwhitneyu
import krippendorff
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from metrics import sentences

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
DATA = json.load(open(os.path.join(HERE, "golden_dataset_v2.json"))); BYID = {it["id"]: it for it in DATA}
DIMS = list("COSTAR"); CATS = ["IDI", "ISA", "IMP", "IRO", "REF", "REG", "FTA", "EMP", "CUL", "MM"]
rng = np.random.default_rng(20261109); S = {}

def jl(f):
    p = os.path.join(RES, f); return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def fprag(sc): return float(np.mean([(sc[d] - 1) / 4 for d in DIMS]))
def r3(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)

def contrast_df(tab, col, higher=True, kinds=("B", "C")):
    t = tab.set_index(["item", "cand"])[col]; rows = []
    for item in tab.item.unique():
        if (item, "A") not in t.index or pd.isna(t[(item, "A")]): continue
        a = t[(item, "A")]
        for k in kinds:
            if (item, k) not in t.index or pd.isna(t[(item, k)]): continue
            x = t[(item, k)]; a2, x2 = (a, x) if higher else (-a, -x)
            rows.append(dict(item=item, kind=k, cat=BYID[item]["category"], origin=BYID[item]["origin"],
                             credit=1.0 if a2 > x2 else (0.5 if a2 == x2 else 0.0), strict=int(a2 > x2), tie=int(a2 == x2), sign=int(np.sign(a2 - x2))))
    return pd.DataFrame(rows)

def boot(d, B=10000):
    if d.empty: return [None, None]
    items = d.item.unique(); g = {i: d[d.item == i].credit.values for i in items}
    v = [np.concatenate([g[i] for i in rng.choice(items, len(items))]).mean() for _ in range(B)]
    return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]

def acc(tab, col, higher=True, ci=True):
    out = {}
    for lab, kinds in [("all", ("B", "C")), ("AB", ("B",)), ("AC", ("C",))]:
        d = contrast_df(tab, col, higher, kinds)
        if d.empty: continue
        out[lab] = dict(n=len(d), acc=r3(d.credit.mean()), strict=r3(d.strict.mean()), ties=r3(d.tie.mean()), ci=boot(d) if ci else None)
    return out

def acc_by(tab, col, higher=True, by="cat"):
    d = contrast_df(tab, col, higher)
    if d.empty: return {}
    return {k: dict(all=r3(g.credit.mean()), AB=r3(g[g.kind == "B"].credit.mean()), AC=r3(g[g.kind == "C"].credit.mean()), n=len(g)) for k, g in d.groupby(by)}

def mcn(tab1, c1, tab2, c2, h1=True, h2=True, kinds=("B", "C")):
    m = contrast_df(tab1, c1, h1, kinds).merge(contrast_df(tab2, c2, h2, kinds), on=["item", "kind"], suffixes=("_1", "_2"))
    b = int(((m.strict_1 == 1) & (m.strict_2 == 0)).sum()); c = int(((m.strict_1 == 0) & (m.strict_2 == 1)).sum())
    return dict(n=len(m), only_first=b, only_second=c, p=float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0)

def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(order): run = max(run, min(1, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj

def kalpha(arr, level):
    a = np.array(arr, dtype=float)
    return r3(krippendorff.alpha(reliability_data=a, level_of_measurement=level)) if a.shape[1] > 1 else None
def cliffs(x, y):
    x, y = np.asarray(x), np.asarray(y); gt = sum((xi > y).sum() for xi in x); lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))

# ================================================================== variant table: judges + metrics
def judge_table(rows, prefix):
    out = []
    for r in rows:
        sc = r["scores"]; cond = r["condition"]; rec = dict(item=r["item"], cand=r["cand"])
        if "score" in sc: rec[f"{prefix}_{cond}"] = (sc["score"] - 1) / 9
        else:
            rec[f"{prefix}_{cond}"] = fprag(sc)
            for d in DIMS: rec[f"{prefix}_{cond}_{d}"] = sc[d]
        out.append(rec)
    if not out: return pd.DataFrame(columns=["item", "cand"])
    return pd.DataFrame(out).groupby(["item", "cand"]).first().reset_index()

T = pd.DataFrame([dict(item=it["id"], cand=k, category=it["category"], origin=it["origin"]) for it in DATA for k in "ABC"])
c1 = [r for r in jl("contrast_j1.jsonl") if r["item"] in BYID]; c2 = [r for r in jl("contrast_j2.jsonl") if r["item"] in BYID]
T = T.merge(judge_table(c1, "J1"), on=["item", "cand"], how="left").merge(judge_table(c2, "J2"), on=["item", "cand"], how="left")
MP = os.path.join(RES, "metrics_contrast.json")
METRICS = [("bleu", "BLEU", True), ("chrf", "chrF", True), ("rouge1", "ROUGE-1", True), ("rouge2", "ROUGE-2", True), ("rougeL", "ROUGE-L", True),
           ("meteor", "METEOR", True), ("ter", "TER", False), ("bertscore", "BERTScore", True), ("sbert", "Sentence-BERT", True),
           ("bartscore", "BARTScore", True), ("nli_ent", "NLI entailment", True), ("nli_bi", "NLI bidirectional", True),
           ("nli_e_minus_c", "NLI entailment minus contradiction", True), ("gpt2_ppl", "GPT-2 perplexity", False), ("fre", "Flesch Reading Ease", True),
           ("n_words", "Length (words)", True), ("f_syntax", "f_syntax (CoLA)", True)]
if os.path.exists(MP):
    M = pd.DataFrame(json.load(open(MP)))
    T = T.merge(M.drop(columns=["category", "origin"]), on=["item", "cand"], how="left")
if "J1_costar" in T and "f_syntax" in T:
    T["M_eval"] = 0.2 * T.f_syntax + 0.8 * T.J1_costar
    T["J1_gate"] = np.minimum(T.J1_costar, (T.J1_costar_O - 1) / 4)
T.to_csv(os.path.join(RES, "variant_table_v2.csv"), index=False)
S["n_items"] = len(DATA); S["n_variants"] = len(T)
S["calls_logged"] = dict(contrast_j1=len(c1), contrast_j2=len(c2))

# ================================================================== design check (Table 3.4)
if "bleu" in T:
    S["design_check"] = {v: {k: r3(T[T.cand == v][k].mean()) for k in ["bleu", "chrf", "rougeL", "bertscore", "n_words", "f_syntax"]} for v in "ABC"}
    S["design_check_new"] = {v: {k: r3(T[(T.cand == v) & (T.origin == "new")][k].mean()) for k in ["bleu", "chrf", "n_words"]} for v in "ABC"}

# ================================================================== Experiment 1: metrics
if "bleu" in T:
    E1 = {}
    for col, lab, hi in METRICS: E1[col] = dict(label=lab, acc=acc(T, col, hi), by_cat=acc_by(T, col, hi))
    E1["clipscore_MM"] = acc(T[T.category == "MM"], "clipscore") if "clipscore" in T else None
    E1["variant_means"] = {col: {v: r3(T[T.cand == v][col].mean()) for v in "ABC"} for col, _, _ in METRICS + [("nli_con", "", True)] if col in T}
    if "J1_costar" in T:
        rows = []; ps = []
        for col, lab, hi in METRICS:
            m = mcn(T, "J1_costar", T, col, True, hi); rows.append((col, m)); ps.append(m["p"])
        adj = holm(ps); E1["mcnemar_J1_vs"] = {col: dict(**m, p_holm=float(a)) for (col, m), a in zip(rows, adj)}
    S["exp1"] = E1

# ================================================================== Experiment 2: J1 prompt conditions
if "J1_costar" in T:
    E2 = {}
    for col in ["J1_plain", "J1_spec_holistic", "J1_costar", "M_eval", "J1_gate", "J1_costar_O"]:
        if col in T: E2[col] = dict(acc=acc(T, col), by_cat=acc_by(T, col), by_origin=acc_by(T, col, by="origin"))
    E2["mcnemar"] = {"costar_vs_spec": mcn(T, "J1_costar", T, "J1_spec_holistic"), "costar_vs_plain": mcn(T, "J1_costar", T, "J1_plain"),
                     "spec_vs_plain": mcn(T, "J1_spec_holistic", T, "J1_plain")}
    if "M_eval" in T: E2["mcnemar"]["meval_vs_costar"] = mcn(T, "M_eval", T, "J1_costar")
    for o in ["pilot", "pilot-heldout", "new"]:
        sub = T[T.origin == o]
        if len(sub): E2[f"origin_{o}"] = {c: acc(sub, c, ci=False).get("all", {}).get("acc") for c in ["J1_plain", "J1_spec_holistic", "J1_costar"]}
    post = T[T.origin != "pilot"]; dev = T[T.origin == "pilot"]
    E2["dev_vs_post"] = {c: dict(dev=acc(dev, c).get("all"), post=acc(post, c).get("all")) for c in ["J1_plain", "J1_spec_holistic", "J1_costar"]}
    if "f_syntax" in T:
        lam = {}
        for l2 in np.round(np.arange(0, 1.01, 0.1), 1):
            T["_m"] = (1 - l2) * T.f_syntax + l2 * T.J1_costar; lam[str(l2)] = acc(T, "_m", ci=False)["all"]["acc"]
        E2["lambda_curve"] = lam; T.drop(columns="_m", inplace=True)
    E2["dims_acc"] = {d: acc(T, f"J1_costar_{d}", ci=False) for d in DIMS}
    E2["profiles"] = {v: {d: r3(T[T.cand == v][f"J1_costar_{d}"].mean()) for d in DIMS} for v in "ABC"}
    E2["profiles_by_cat"] = {c: {v: {d: r3(T[(T.cand == v) & (T.category == c)][f"J1_costar_{d}"].mean()) for d in DIMS} for v in "ABC"} for c in CATS}
    C = T[T.cand == "C"]; B = T[T.cand == "B"]
    E2["halo"] = dict(C_O_le2=int((C.J1_costar_O <= 2).sum()), C_n=len(C), B_S_ge4=int((B.J1_costar_S >= 4).sum()), B_O_le2=int((B.J1_costar_O <= 2).sum()), B_n=len(B))
    tx = T[T.category != "MM"]
    E2["dim_corr_text"] = {f"{a}{b}": r3(spearmanr(tx[f"J1_costar_{a}"], tx[f"J1_costar_{b}"]).correlation) for i, a in enumerate(DIMS) for b in DIMS[i + 1:]}
    E2["score_means"] = {c: {v: r3(T[T.cand == v][c].mean()) for v in "ABC"} for c in ["J1_plain", "J1_spec_holistic", "J1_costar"]}
    # errors of the CO-STAR evaluator
    d = contrast_df(T, "J1_costar"); E2["errors"] = d[d.credit < 1][["item", "kind", "credit"]].to_dict("records")
    # Which dimensions produce the errors: for each misordered or tied contrast, the dimensions on which A scored lower than the failed variant
    TI = T.set_index(["item", "cand"]); cnt = {k: 0 for k in DIMS}; a_low = {k: 0 for k in DIMS}
    for e in E2["errors"]:
        for k in DIMS:
            if TI.loc[(e["item"], "A"), f"J1_costar_{k}"] < TI.loc[(e["item"], e["kind"]), f"J1_costar_{k}"]: cnt[k] += 1
    A = T[T.cand == "A"]
    E2["error_dims"] = dict(n_errors=len(E2["errors"]), A_lower_on=cnt,
                            A_dim_le2={k: int((A[f"J1_costar_{k}"] <= 2).sum()) for k in DIMS}, A_n=len(A),
                            A_R_le2_by_cat={c: int((A[A.category == c].J1_costar_R <= 2).sum()) for c in CATS},
                            A_O_le2_items=A[A.J1_costar_O <= 2].item.tolist(), A_R_le2_items=A[A.J1_costar_R <= 2].item.tolist(),
                            items_with_error=sorted(set(e["item"] for e in E2["errors"])))
    if "J2_costar_R" in T:
        A2 = T[(T.cand == "A") & T.J2_costar_R.notna()]
        E2["error_dims"]["J2_A_dim_le2"] = {k: int((A2[f"J2_costar_{k}"] <= 2).sum()) for k in DIMS}; E2["error_dims"]["J2_A_n"] = len(A2)
    S["exp2"] = E2

# ================================================================== Experiment 3: charts without key
if "J1_costar_nokey" in T:
    mm = T[T.category == "MM"]
    S["exp3"] = dict(key=acc(mm, "J1_costar"), nokey=acc(mm, "J1_costar_nokey"), mcnemar=mcn(mm, "J1_costar", mm, "J1_costar_nokey"),
                     O_means={c: {v: r3(mm[mm.cand == v][f"{c}_O"].mean()) for v in "ABC"} for c in ["J1_costar", "J1_costar_nokey"]},
                     by_origin={o: dict(key=acc(mm[mm.origin == o], "J1_costar", ci=False).get("all", {}).get("acc"), nokey=acc(mm[mm.origin == o], "J1_costar_nokey", ci=False).get("all", {}).get("acc")) for o in ["pilot", "new"]},
                     spec=acc(mm, "J1_spec_holistic"), plain=acc(mm, "J1_plain"))
    d = contrast_df(mm, "J1_costar_nokey"); S["exp3"]["errors_nokey"] = d[d.credit < 1][["item", "kind", "credit"]].to_dict("records")

# ================================================================== Experiment 4: J2, agreement, pairwise, robustness
tx = T[T.category != "MM"]
if "J2_costar" in T and T.J2_costar.notna().sum() > 30:
    E4 = {}
    for col in ["J2_plain", "J2_spec_holistic", "J2_costar"]:
        if col in T: E4[col] = dict(acc=acc(tx, col), by_cat=acc_by(tx, col))
    E4["J1_on_text"] = {c: acc(tx, c, ci=False)["all"]["acc"] for c in ["J1_plain", "J1_spec_holistic", "J1_costar"]}
    E4["mcnemar"] = {"J2_costar_vs_spec": mcn(tx, "J2_costar", tx, "J2_spec_holistic"), "J2_spec_vs_plain": mcn(tx, "J2_spec_holistic", tx, "J2_plain"),
                     "J2_costar_vs_plain": mcn(tx, "J2_costar", tx, "J2_plain"), "J1_vs_J2_costar": mcn(tx, "J1_costar", tx, "J2_costar")}
    both = tx.dropna(subset=["J1_costar", "J2_costar"])
    E4["agreement"] = dict(n=len(both), alpha_fprag=kalpha([both.J1_costar, both.J2_costar], "interval"), rho_fprag=r3(spearmanr(both.J1_costar, both.J2_costar).correlation),
                           dims={d: dict(alpha_ord=kalpha([both[f"J1_costar_{d}"], both[f"J2_costar_{d}"]], "ordinal"),
                                         qwk=r3(cohen_kappa_score(both[f"J1_costar_{d}"].astype(int), both[f"J2_costar_{d}"].astype(int), weights="quadratic")),
                                         exact=r3((both[f"J1_costar_{d}"] == both[f"J2_costar_{d}"]).mean())) for d in DIMS})
    E4["J2_profiles"] = {v: {d: r3(tx[tx.cand == v][f"J2_costar_{d}"].mean()) for d in DIMS} for v in "ABC"}
    E4["J2_score_means"] = {c: {v: r3(tx[tx.cand == v][c].mean()) for v in "ABC"} for c in ["J2_plain", "J2_spec_holistic", "J2_costar"] if c in tx}
    tx2 = tx.copy(); tx2["jury"] = (tx2.J1_costar + tx2.J2_costar) / 2; E4["jury"] = acc(tx2, "jury")
    E4["J2_C_O_le2"] = int((tx[tx.cand == "C"].J2_costar_O <= 2).sum())
    S["exp4"] = E4
P = jl("pairwise_j1.jsonl")
if P:
    pdf = pd.DataFrame(P); pdf = pdf[pdf.item.isin(BYID)]
    piv = pdf.pivot_table(index=["item", "pair"], columns="order", values="a_wins", aggfunc="first").dropna()
    S.setdefault("exp4", {})["pairwise"] = dict(n_pairs=len(piv), n_calls=len(pdf), a_first=r3(piv["A-first"].mean()), a_second=r3(piv["A-second"].mean()),
        consistent=r3((piv["A-first"] == piv["A-second"]).mean()), both_correct=r3((piv["A-first"] & piv["A-second"]).mean()),
        first_pos_chosen=r3(np.mean([(r["choice"] == 1) for r in pdf.to_dict("records")])),
        by_pair={p: dict(n=len(g), a_first=r3(g["A-first"].mean()), a_second=r3(g["A-second"].mean()), consistent=r3((g["A-first"] == g["A-second"]).mean()), both_correct=r3((g["A-first"] & g["A-second"]).mean())) for p, g in piv.groupby(level="pair")})
    PW = S["exp4"]["pairwise"]
    pv = piv.reset_index(); pv["cat"] = pv.item.map(lambda i: BYID[i]["category"]); pv["both"] = pv["A-first"] & pv["A-second"]
    PW["by_cat"] = {c: dict(n=len(g), both_correct=r3(g.both.mean()), consistent=r3((g["A-first"] == g["A-second"]).mean())) for c, g in pv.groupby("cat")}
    PW["inconsistent_favour_first"] = int(((pv["A-first"]) & (~pv["A-second"])).sum()); PW["inconsistent_favour_second"] = int(((~pv["A-first"]) & (pv["A-second"])).sum())
    PW["both_wrong"] = int(((~pv["A-first"]) & (~pv["A-second"])).sum())
    pt = contrast_df(tx, "J1_costar").rename(columns={"kind": "k"}); pt["pair"] = "A" + pt.k
    mm = pv.merge(pt[["item", "pair", "strict"]], on=["item", "pair"], how="inner")
    b, c = int((mm.both & (mm.strict == 0)).sum()), int(((~mm.both) & (mm.strict == 1)).sum())
    PW["vs_costar"] = dict(n=len(mm), pair_only=b, costar_only=c, p=float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0, costar_strict=r3(mm.strict.mean()))
R = [r for r in jl("robust_j1.jsonl") if r["item"] in BYID]
SUBP = os.path.join(RES, "robust_subset.json")
if R and os.path.exists(SUBP):
    sub = set(json.load(open(SUBP))); Rt = judge_table([r for r in R if r["item"] in sub], "J1")
    X = T[T.item.isin(sub)][["item", "cand", "category", "origin", "J1_costar"] + [f"J1_costar_{d}" for d in DIMS]].merge(Rt, on=["item", "cand"], how="left")
    E = {"n_items": len(sub), "reference": acc(X, "J1_costar")}
    ref_sign = contrast_df(X, "J1_costar").set_index(["item", "kind"]).sign
    conds = [c for c in ["costar_para", "costar_s1", "costar_s2", "costar_s3"] if f"J1_{c}" in X and X[f"J1_{c}"].notna().all()]
    for c in conds:
        col = f"J1_{c}"; sg = contrast_df(X, col).set_index(["item", "kind"]).sign
        E[c] = dict(acc=acc(X, col), alpha=kalpha([X.J1_costar, X[col]], "interval"), rho=r3(spearmanr(X.J1_costar, X[col]).correlation),
                    flips=int((sg != ref_sign.loc[sg.index]).sum()), exact_dims={d: r3((X[f"J1_costar_{d}"] == X[f"{col}_{d}"]).mean()) for d in DIMS})
    ss = [c for c in conds if c.startswith("costar_s")]
    if len(ss) == 3:
        X["mean3"] = X[[f"J1_{c}" for c in ss]].mean(axis=1); E["mean_of_3"] = acc(X, "mean3")
        E["alpha_3runs"] = kalpha([X[f"J1_{c}"] for c in ss], "interval")
        E["identical_vectors"] = r3(np.mean([all(tuple(X.loc[i, f"J1_{c}_{d}"] for d in DIMS) == tuple(X.loc[i, f"J1_{ss[0]}_{d}"] for d in DIMS) for c in ss) for i in X.index]))
        signs = [contrast_df(X, f"J1_{c}").set_index(["item", "kind"]).sign for c in ss]
        E["sign_agreement_3"] = r3(np.mean((signs[0] == signs[1]) & (signs[1] == signs[2])))
    S.setdefault("exp4", {})["robustness"] = E

# ================================================================== Experiment 5: generators
gens = {g: {r["item"]: r for r in jl(f"gen_{g}.jsonl") if r["item"] in BYID} for g in ["qwen3vl-2b", "smolvlm2-2.2b"]}
GJ1 = {(r["generator"], r["item"]): r["scores"] for r in jl("genjudge_j1.jsonl") if r["item"] in BYID}
GJ2 = {(r["generator"], r["item"]): r["scores"] for r in jl("genjudge_j2.jsonl") if r["item"] in BYID}
GP = os.path.join(RES, "metrics_gen.json"); GM = {(r["generator"], r["item"]): r for r in json.load(open(GP))} if os.path.exists(GP) else {}

def clean_lines(text):
    t = re.sub(r"\*\*|__|`", "", text); t = re.sub(r"(?m)^\s*#+\s*", "", t); t = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", "", t)
    return [l.strip() for l in t.split("\n") if l.strip()]
SAL = re.compile(r"^(dear|hi|hello|hey|good (morning|afternoon|evening))\b[^.!?]*,?$", re.I)
OFF = re.compile(r"^(?:(best|kind|warm|warmest)?\s*(regards|wishes|thanks|thank you|sincerely|cheers)|best|yours(?: (?:sincerely|truly|faithfully))?)\b[^.!?]*,?$", re.I)
ABBR = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\.", re.I)
LEAD = re.compile(r"^(?:(?:dear|hi|hello|hey)\b[^.!?,]{0,40}[,!]|good (?:morning|afternoon|evening)\b[^.!?,]{0,40}[,!])\s*", re.I)
def n_sent(text):
    body = [l for l in clean_lines(text) if not SAL.match(l) and not OFF.match(l) and not re.match(r"^\[?[A-Z][a-z]+( [A-Z][a-z]+)?\]?$", l)]
    if not body: return 0
    t = ABBR.sub(lambda m: m.group(1), " ".join(body)); t = LEAD.sub("", ABBR.sub(lambda m: m.group(1), body[0])) + t[len(ABBR.sub(lambda m: m.group(1), body[0])):]
    return len(sentences(t)) if t.strip() else 0
def fmt_ok(text, spec):
    s = spec.lower(); n = n_sent(text); w = len(" ".join(clean_lines(text)).split())
    if "at most 60 words" in s: ok = w <= 60
    elif "followed by" in s: ok = 2 <= n <= 3
    elif "exactly two" in s or s.startswith("two sentences") or "of exactly two" in s: ok = n == 2
    elif "two or three" in s: ok = 2 <= n <= 3
    elif "three sentences or fewer" in s: ok = 1 <= n <= 3
    elif s.startswith("three sentences"): ok = n == 3
    elif "one or two" in s: ok = 1 <= n <= 2
    elif "short direct answer" in s: ok = n <= 2 and w <= 40
    elif s.startswith("the rewritten sentence") or s.startswith("the shortened sentence"): ok = 1 <= n <= 2
    elif "sentence" in s and s.split()[0] in ("one",): ok = n == 1
    else: ok = True
    if "salutation" in s: ok = ok and bool(clean_lines(text)) and bool(re.match(r"^(dear|hi|hello)\b", clean_lines(text)[0], re.I))
    return bool(ok)
FACTS = {"MM-01": [r"\bapril|\bapr\b", r"\b31\b|31,000|31 ?k"], "MM-02": [r"\brent\b", r"\b40\s?%|40 per ?cent"], "MM-03": [r"telugu", r"\b9\b|nine"],
         "MM-04": [r"saturday|\bsat\b", r"\b55\b"], "MM-05": [r"\b30\s?%", r"\b55\s?%", r"online"], "MM-06": [r"\b95\b", r"\b2 hours|\(2,|two hours|\b2\b"],
         "MM-07": [r"august|\baug\b", r"\b185\b"], "MM-08": [r"\b60\b", r"week 4|\bw4\b|fourth week"],
         "MM-09": [r"\bapril|\bapr\b", r"2,?150"], "MM-10": [r"class 9\b|class nine|\b9\b", r"\b60\b"], "MM-11": [r"cycl", r"\b12\s?(%|per ?cent)"],
         "MM-12": [r"plastic", r"cloth", r"\b150\b", r"\b380\b"], "MM-13": [r"temperat|hott|warm", r"ris|increas|go(es)? up|climb|higher|more"],
         "MM-14": [r"dinosaur", r"\b850\b"], "MM-15": [r"\b8\s?(am|a\.m\.|o'?clock)|8:00|\beight\b", r"\b210\b"], "MM-16": [r"\b92\b", r"\b65\b", r"\b27\b"]}
def facts_ok(item, text): return all(re.search(p, " ".join(clean_lines(text)).lower()) for p in FACTS[item])
T["fmt_ok"] = [fmt_ok(BYID[i]["candidates"][c], BYID[i]["costar"]["response_format"]) for i, c in zip(T.item, T.cand)]
S["format_rule_on_variants"] = {v: r3(T[T.cand == v].fmt_ok.mean()) for v in "ABC"}
if "exp2" in S:   # exploratory: the judge's Response format score replaced by the rule-based format check (1 if the rule passes, 0 otherwise)
    T["_hyb"] = (sum((T[f"J1_costar_{k}"] - 1) / 4 for k in "COSTA") + T.fmt_ok.astype(float)) / 6
    S["exp2"]["hybrid_rule_R"] = dict(acc=acc(T, "_hyb"), by_cat=acc_by(T, "_hyb")); T.drop(columns="_hyb", inplace=True)
G = []
for g, d in gens.items():
    for i, r in d.items():
        rec = dict(generator=g, item=i, category=BYID[i]["category"], origin=BYID[i]["origin"], output=r["output"], finish=r["finish"], latency=r["latency"],
                   n_words=len(r["output"].split()), fmt_ok=fmt_ok(r["output"], BYID[i]["costar"]["response_format"]))
        if BYID[i]["category"] == "MM": rec["facts_ok"] = facts_ok(i, r["output"])
        if (g, i) in GJ1:
            rec["J1"] = fprag(GJ1[(g, i)]); rec.update({f"J1_{d}": GJ1[(g, i)][d] for d in DIMS})
        if (g, i) in GJ2:
            rec["J2"] = fprag(GJ2[(g, i)]); rec.update({f"J2_{d}": GJ2[(g, i)][d] for d in DIMS})
        if (g, i) in GM: rec.update({k: v for k, v in GM[(g, i)].items() if k not in ("generator", "item", "category", "origin", "n_words")})
        G.append(rec)
G = pd.DataFrame(G)
if len(G):
    if "f_syntax" in G and "J1" in G: G["M_eval"] = 0.2 * G.f_syntax + 0.8 * G.J1
    G.drop(columns=["output"]).to_csv(os.path.join(RES, "gen_table_v2.csv"), index=False)
    E5 = dict(n=G.groupby("generator").size().to_dict())
    cols = [c for c in ["J1", "J2", "M_eval", "f_syntax", "bleu", "chrf", "rougeL", "bertscore", "sbert", "nli_bi", "bartscore", "gpt2_ppl", "n_words"] + [f"J1_{d}" for d in DIMS] + [f"J2_{d}" for d in DIMS] if c in G]
    E5["means"] = {g: {c: r3(d[c].mean()) for c in cols} for g, d in G.groupby("generator")}
    E5["fmt_rate"] = {g: r3(d.fmt_ok.mean()) for g, d in G.groupby("generator")}
    E5["truncated"] = {g: int((d.finish == "length").sum()) for g, d in G.groupby("generator")}
    E5["latency"] = {g: r3(d.latency.mean()) for g, d in G.groupby("generator")}
    if G.generator.nunique() == 2:
        a, b = [G[G.generator == g].set_index("item") for g in ["qwen3vl-2b", "smolvlm2-2.2b"]]
        common = a.index.intersection(b.index); eff = {}
        for c in cols:
            x, y = a.loc[common, c], b.loc[common, c]; ok = x.notna() & y.notna()
            if ok.sum() < 10: continue
            x, y = x[ok], y[ok]
            try: p = float(wilcoxon(x, y).pvalue)
            except ValueError: p = 1.0
            eff[c] = dict(n=int(ok.sum()), g1=r3(x.mean()), g2=r3(y.mean()), g1_med=r3(x.median()), g2_med=r3(y.median()), g1_higher=int((x > y).sum()), g2_higher=int((x < y).sum()), p=p, delta=r3(cliffs(x, y)))
        E5["effects"] = eff
        if "J1" in G: E5["by_cat"] = {c: {g: r3(G[(G.generator == g) & (G.category == c)].J1.mean()) for g in ["qwen3vl-2b", "smolvlm2-2.2b"]} for c in CATS}
        if "J2" in G: E5["by_cat_J2"] = {c: {g: r3(G[(G.generator == g) & (G.category == c)].J2.mean()) for g in ["qwen3vl-2b", "smolvlm2-2.2b"]} for c in CATS if c != "MM"}
    if "J1" in G and "bleu" in G:
        E5["spearman_J1"] = {c: r3(spearmanr(G.J1, G[c], nan_policy="omit").correlation) for c in ["J2", "f_syntax", "bleu", "chrf", "rougeL", "bertscore", "sbert", "nli_bi", "bartscore", "gpt2_ppl", "n_words"] if c in G}
    if "J1" in G and "J2" in G:
        bb = G.dropna(subset=["J1", "J2"])
        E5["agreement"] = dict(n=len(bb), alpha=kalpha([bb.J1, bb.J2], "interval"), rho=r3(spearmanr(bb.J1, bb.J2).correlation),
                               qwk={d: r3(cohen_kappa_score(bb[f"J1_{d}"].astype(int), bb[f"J2_{d}"].astype(int), weights="quadratic")) for d in DIMS})
    if "f_syntax" in G and "J1" in G:
        E5["lambda"] = {str(l2): {g: r3(((1 - l2) * d.f_syntax + l2 * d.J1).mean()) for g, d in G.groupby("generator")} for l2 in np.round(np.arange(0, 1.01, 0.2), 1)}
    if "J1_O" in G:
        G["J1_gate"] = np.minimum(G.J1, (G.J1_O - 1) / 4)
        G["J1_hybrid"] = (G[[f"J1_{d}" for d in "COSTA"]].sum(axis=1) - 5 + np.where(G.fmt_ok, 4, 0)) / 24
        E5["extra_scores"] = {g: dict(gate=r3(d.J1_gate.mean()), hybrid=r3(d.J1_hybrid.mean()), O_le2=int((d.J1_O <= 2).sum()), O_eq5=int((d.J1_O == 5).sum()),
                                      all5=int((d[[f"J1_{x}" for x in DIMS]] == 5).all(axis=1).sum())) for g, d in G.groupby("generator")}
        if G.generator.nunique() == 2:
            a, b = [G[G.generator == g].set_index("item") for g in ["qwen3vl-2b", "smolvlm2-2.2b"]]
            both = a.index[(a.fmt_ok & b.reindex(a.index).fmt_ok).values]
            sub = {}
            for c in ["J1", "J1_O", "J1_gate", "J1_hybrid"]:
                x, y = a.loc[both, c], b.loc[both, c]
                try: p = float(wilcoxon(x, y).pvalue)
                except ValueError: p = 1.0
                sub[c] = dict(n=len(both), g1=r3(x.mean()), g2=r3(y.mean()), g1_higher=int((x > y).sum()), g2_higher=int((x < y).sum()), p=p, delta=r3(cliffs(x, y)))
            E5["both_fmt_ok"] = sub
            for c in ["J1_gate", "J1_hybrid"]:
                x, y = a[c], b.reindex(a.index)[c]
                E5["effects"][c] = dict(n=len(x), g1=r3(x.mean()), g2=r3(y.mean()), g1_higher=int((x > y).sum()), g2_higher=int((x < y).sum()), p=float(wilcoxon(x, y).pvalue), delta=r3(cliffs(x, y)))
            # item-level preference: does a metric prefer the same generator as J1?
            pa = {}
            dj = np.sign(a.J1 - b.reindex(a.index).J1)
            for c in ["J2", "bleu", "chrf", "rougeL", "bertscore", "sbert", "nli_bi", "bartscore", "f_syntax", "n_words"]:
                if c not in G: continue
                dm = np.sign(a[c] - b.reindex(a.index)[c]) * (-1 if c == "n_words" else 1)
                ok = (dj != 0) & (dm != 0) & dm.notna()
                pa[c] = dict(n=int(ok.sum()), agree=r3((dj[ok] == dm[ok]).mean()))
            E5["pref_agree_J1"] = pa
        FEAT = {"subject_line": r"(?im)^\s*\**subject:", "placeholder": r"(?i)\[[^\]]*(name|recipient|colleague|manager|friend)[^\]]*\]", "meta_framing": r"(?i)^\s*(sure[!,.]|here('s| is| are)|certainly)",
                "numbered_list": r"(?m)^\s*\d+\.\s", "bullets": r"(?m)^\s*[-*\u2022]\s"}
        E5["features"] = {g: dict({k: int(d.output.str.contains(p, regex=True).sum()) for k, p in FEAT.items()},
                                  any=int(d.output.apply(lambda o: any(re.search(p, o) for p in FEAT.values())).sum())) for g, d in G.groupby("generator")}
        if "J2" in G and G.J2.notna().sum() > 100:
            bt = both[[BYID[i]["category"] != "MM" for i in both]]
            x, y = a.loc[bt, "J2"], b.loc[bt, "J2"]
            try: p2 = float(wilcoxon(x, y).pvalue)
            except ValueError: p2 = 1.0
            E5["both_fmt_ok"]["J2"] = dict(n=len(bt), g1=r3(x.mean()), g2=r3(y.mean()), g1_higher=int((x > y).sum()), g2_higher=int((x < y).sum()), p=p2)
            gg = G.dropna(subset=["J2"])
            f1, f2 = gg.J1_O <= 2, gg.J2_O <= 2
            E5["misread_flags"] = dict(n=len(gg), both=int((f1 & f2).sum()), j1_only=int((f1 & ~f2).sum()), j2_only=int((~f1 & f2).sum()), neither=int((~f1 & ~f2).sum()),
                                       kappa=r3(cohen_kappa_score(f1, f2)), by_gen={g: dict(j1=int((d.J1_O <= 2).sum()), j2=int((d.J2_O <= 2).sum()), j2_all5=int((d[[f"J2_{x}" for x in DIMS]] == 5).all(axis=1).sum()), j1_all5=int((d[[f"J1_{x}" for x in DIMS]] == 5).all(axis=1).sum())) for g, d in gg.groupby("generator")})
        G.drop(columns=["output"]).to_csv(os.path.join(RES, "gen_table_v2.csv"), index=False)
    def crit(df, flag, score):
        d = df.dropna(subset=[score]); y = d[flag].astype(int)
        out = dict(n=len(d), n_ok=int(y.sum()), mean_ok=r3(d[d[flag]][score].mean()), mean_fail=r3(d[~d[flag]][score].mean()))
        if 0 < y.sum() < len(y): out.update(auc=r3(roc_auc_score(y, d[score])), p=float(mannwhitneyu(d[d[flag]][score], d[~d[flag]][score]).pvalue))
        return out
    CR = {}
    if "J1_R" in G: CR["format_J1_R"] = crit(G, "fmt_ok", "J1_R")
    if "J2_R" in G: CR["format_J2_R"] = crit(G[G.category != "MM"], "fmt_ok", "J2_R")
    if "J1_O" in G and "facts_ok" in G: CR["facts_J1_O"] = crit(G[G.category == "MM"].astype({"facts_ok": bool}), "facts_ok", "J1_O")
    if "J1_costar_R" in T: CR["format_variants_J1_R"] = crit(T, "fmt_ok", "J1_costar_R")
    if "J2_costar_R" in T: CR["format_variants_J2_R"] = crit(T[T.category != "MM"].dropna(subset=["J2_costar_R"]), "fmt_ok", "J2_costar_R")
    E5["criterion"] = CR
    S["exp5"] = E5

# ================================================================== throughput
calls = []
for f in ["contrast_j1.jsonl", "contrast_j2.jsonl", "pairwise_j1.jsonl", "genjudge_j1.jsonl", "genjudge_j2.jsonl", "robust_j1.jsonl"]:
    for r in jl(f):
        if r["item"] in BYID and (f != "robust_j1.jsonl" or not os.path.exists(SUBP) or r["item"] in set(json.load(open(SUBP)))):
            calls.append(dict(file=f, reused=bool(r.get("reused_from_pilot")), latency=r["latency"], valid=bool(r.get("raw")), cond=re.sub(r"^costar_s\d$", "costar_sampled", r.get("condition", "pair"))))
gc = [dict(gen=gname, latency=r["latency"], reused=bool(r.get("reused_from_pilot")), tokens=(r.get("usage") or {}).get("completion_tokens")) for gname, g in gens.items() for r in g.values()]
C = pd.DataFrame(calls)
if len(C):
    GC = pd.DataFrame(gc)
    S["throughput"] = dict(eval_calls=len(C), eval_reused=int(C.reused.sum()), eval_hours=r3(C.latency.sum() / 3600),
                           eval_new_hours=r3(C[~C.reused].latency.sum() / 3600), gen_calls=len(gc), gen_hours=r3(sum(x["latency"] for x in gc) / 3600),
                           eval_valid=int(C.valid.sum()), total_calls=len(C) + len(gc), total_reused=int(C.reused.sum()) + int(GC.reused.sum()),
                           total_hours=r3((C.latency.sum() + GC.latency.sum()) / 3600), total_new_hours=r3((C[~C.reused].latency.sum() + GC[~GC.reused].latency.sum()) / 3600),
                           mm_costar_median_s=r3(np.median([r["latency"] for r in jl("contrast_j1.jsonl") if r.get("condition") == "costar" and r["item"].startswith("MM")])),
                           text_costar_median_s=r3(np.median([r["latency"] for r in jl("contrast_j1.jsonl") if r.get("condition") == "costar" and not r["item"].startswith("MM")])),
                           by_file={f: dict(n=len(d), mean_s=r3(d.latency.mean()), hours=r3(d.latency.sum() / 3600)) for f, d in C.groupby("file")},
                           by_file_cond={f"{f}|{c}": dict(n=len(d), reused=int(d.reused.sum()), mean_s=r3(d.latency.mean()), median_s=r3(d.latency.median()),
                                                          hours=r3(d.latency.sum() / 3600)) for (f, c), d in C.groupby(["file", "cond"])},
                           by_gen={g: dict(n=len(d), reused=int(d.reused.sum()), mean_s=r3(d.latency.mean()), median_s=r3(d.latency.median()), hours=r3(d.latency.sum() / 3600), mean_tokens=r3(d.tokens.mean())) for g, d in GC.groupby("gen")})
json.dump(S, open(os.path.join(RES, "summary_v2.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("summary_v2 written:", list(S.keys()))
