"""Reference-based overlap metrics and the reference-free syntactic score f_syntax.

f_syntax(x): mean probability of grammatical acceptability over the sentences of x, from a
RoBERTa classifier fine-tuned on CoLA (Warstadt et al., 2019; checkpoint textattack/roberta-base-CoLA).
"""
import json, os, re, sys
import torch
torch.set_num_threads(1)
import sacrebleu
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
_rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_tok = _mdl = None

def strip_markdown(text):
    """Remove Markdown emphasis, heading and list markers so that metrics see plain prose."""
    t = re.sub(r"\*\*|__|`", "", text)
    t = re.sub(r"(?m)^\s*#+\s*", "", t)
    t = re.sub(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s+", "", t)
    t = re.sub(r"\s*\n+\s*", " ", t)
    return t.strip()

def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]

def f_syntax(text):
    global _tok, _mdl
    if _mdl is None:
        _tok = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
        _mdl = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA").eval()
    sents = sentences(text) or [text]
    with torch.no_grad():
        enc = _tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=128)
        p = torch.softmax(_mdl(**enc).logits, -1)[:, 1]   # label 1 = acceptable
    return float(p.mean())

def overlap(cands, refs):
    from bert_score import score as bscore
    out = []
    for c, r in zip(cands, refs):
        out.append(dict(bleu=sacrebleu.sentence_bleu(c, [r]).score,
                        chrf=sacrebleu.sentence_chrf(c, [r]).score,
                        rougeL=_rs.score(r, c)["rougeL"].fmeasure * 100))
    P, R, F = bscore(cands, refs, lang="en", model_type="roberta-large", verbose=False, batch_size=16)
    for o, f in zip(out, F.tolist()): o["bertscore"] = f * 100
    return out

if __name__ == "__main__":
    what = sys.argv[1]
    data = json.load(open(os.path.join(HERE, "golden_dataset.json")))
    if what == "sanity":
        for s in ["The results indicate that the chatbot frequently misinterprets sarcastic utterances.", "Results the chatbot misinterprets frequently indicate utterances the sarcastic.", "lol yeah the post is a mess rn."]:
            print(round(f_syntax(s), 3), s)
    if what == "contrast":
        rows, cands, refs = [], [], []
        for it in data:
            for k in "ABC":
                rows.append(dict(item=it["id"], category=it["category"], cand=k)); cands.append(it["candidates"][k]); refs.append(it["reference"])
        ov = overlap(cands, refs)
        for row, o, c in zip(rows, ov, cands): row.update(o); row["f_syntax"] = f_syntax(c); row["n_words"] = len(c.split())
        json.dump(rows, open(os.path.join(HERE, "results", "contrast_metrics.json"), "w"), indent=1)
        print("wrote", len(rows))
    if what == "gen":
        byid = {it["id"]: it for it in data}
        rows, cands, refs = [], [], []
        for f in ["gen_qwen3vl-2b.jsonl", "gen_smolvlm2-2.2b.jsonl"]:
            for l in open(os.path.join(HERE, "results", f)):
                g = json.loads(l); rows.append(dict(item=g["item"], category=g["category"], generator=g["generator"]))
                cands.append(strip_markdown(g["output"]) if g["output"].strip() else "(empty)"); refs.append(byid[g["item"]]["reference"])
        ov = overlap(cands, refs)
        for row, o, c in zip(rows, ov, cands): row.update(o); row["f_syntax"] = f_syntax(c); row["n_words"] = len(c.split())
        json.dump(rows, open(os.path.join(HERE, "results", "gen_metrics.json"), "w"), indent=1)
        print("wrote", len(rows))
