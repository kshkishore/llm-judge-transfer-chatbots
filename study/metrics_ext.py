"""Extended metric suite (Chapter 3, Section 3.4.6): further reference-based and reference-free scores.

Reference-based:  ROUGE-1, ROUGE-2, METEOR, TER, BARTScore (bart-large-cnn), sentence-embedding cosine
                  (all-mpnet-base-v2), NLI entailment / contradiction (roberta-large-mnli)
Reference-free:   GPT-2 perplexity, Flesch Reading Ease, Flesch-Kincaid grade, CLIPScore (chart items only)
Usage: python metrics_ext.py <contrast|gen|star|ext>
"""
import json, os, re, sys, math
import torch
torch.set_num_threads(1)
import sacrebleu, textstat
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from metrics import strip_markdown, sentences

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
_rs = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
_ter = sacrebleu.metrics.TER()
tokens = lambda s: re.findall(r"[a-z0-9]+(?:'[a-z]+)?", s.lower())

def lexical(c, r):
    s = _rs.score(r, c)
    return dict(rouge1=s["rouge1"].fmeasure * 100, rouge2=s["rouge2"].fmeasure * 100,
                meteor=meteor_score([tokens(r)], tokens(c)) * 100, ter=_ter.sentence_score(c, [r]).score,
                fre=textstat.flesch_reading_ease(c), fkgl=textstat.flesch_kincaid_grade(c), n_sents=len(sentences(c)))

def bartscore(cands, refs, name="facebook/bart-large-cnn"):
    tok = AutoTokenizer.from_pretrained(name); mdl = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
    def ll(src, tgt):   # mean log-likelihood per target token of tgt given src
        e = tok(src, return_tensors="pt", truncation=True, max_length=512); t = tok(tgt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad(): return -float(mdl(input_ids=e.input_ids, attention_mask=e.attention_mask, labels=t.input_ids).loss)
    out = []
    for c, r in zip(cands, refs):
        p, rc = ll(r, c), ll(c, r); out.append(dict(bart_p=p, bart_r=rc, bartscore=(p + rc) / 2))
    del mdl; return out

def sbert(cands, refs, name="sentence-transformers/all-mpnet-base-v2"):
    tok = AutoTokenizer.from_pretrained(name); mdl = AutoModel.from_pretrained(name).eval()
    def emb(texts):
        e = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=384)
        with torch.no_grad(): h = mdl(**e).last_hidden_state
        m = e.attention_mask.unsqueeze(-1).float(); v = (h * m).sum(1) / m.sum(1)
        return torch.nn.functional.normalize(v, dim=-1)
    out = []
    for i in range(0, len(cands), 8):
        a, b = emb(cands[i:i + 8]), emb(refs[i:i + 8]); out += [dict(sbert=float(x) * 100) for x in (a * b).sum(-1)]
    del mdl; return out

def nli(cands, refs, name="FacebookAI/roberta-large-mnli"):
    tok = AutoTokenizer.from_pretrained(name); mdl = AutoModelForSequenceClassification.from_pretrained(name).eval()
    lab = {v.lower(): k for k, v in mdl.config.id2label.items()}; E, C = lab["entailment"], lab["contradiction"]
    def pr(premise, hyp):
        e = tok(premise, hyp, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad(): p = torch.softmax(mdl(**e).logits, -1)[0]
        return float(p[E]), float(p[C])
    out = []
    for c, r in zip(cands, refs):
        e1, c1 = pr(r, c); e2, c2 = pr(c, r)
        out.append(dict(nli_ent=e1 * 100, nli_con=c1 * 100, nli_ent_rev=e2 * 100, nli_con_rev=c2 * 100, nli_bi=(e1 + e2) / 2 * 100, nli_e_minus_c=(e1 - c1) * 100))
    del mdl; return out

def gpt2_ppl(cands, name="openai-community/gpt2"):
    tok = AutoTokenizer.from_pretrained(name); mdl = AutoModelForCausalLM.from_pretrained(name).eval()
    out = []
    for c in cands:
        ids = tok(tok.bos_token + c, return_tensors="pt", truncation=True, max_length=512).input_ids
        with torch.no_grad(): nll = float(mdl(ids, labels=ids).loss)
        out.append(dict(gpt2_nll=nll, gpt2_ppl=math.exp(nll)))
    del mdl; return out

def clipscore(cands, images, name="openai/clip-vit-base-patch32"):
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    mdl = CLIPModel.from_pretrained(name).eval(); proc = CLIPProcessor.from_pretrained(name)
    out = []
    for c, im in zip(cands, images):
        if not im: out.append({}); continue
        e = proc(text=[c], images=Image.open(os.path.join(HERE, "images", im)).convert("RGB"), return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad(): o = mdl(**e)
        cos = float(torch.nn.functional.cosine_similarity(o.image_embeds, o.text_embeds)[0])
        out.append(dict(clip_cos=cos, clipscore=2.5 * max(cos, 0.0)))     # CLIP-S of Hessel et al. (2021), w = 2.5
    del mdl; return out

def run(rows, cands, refs, images, out_name):
    for row, c, r in zip(rows, cands, refs): row.update(lexical(c, r)); row["n_words"] = len(c.split())
    for fn, args in [(sbert, (cands, refs)), (gpt2_ppl, (cands,)), (nli, (cands, refs)), (bartscore, (cands, refs))]:
        print("running", fn.__name__, flush=True)
        for row, o in zip(rows, fn(*args)): row.update(o)
        json.dump(rows, open(os.path.join(RES, out_name), "w"), indent=1)
    if any(images):
        print("running clipscore", flush=True)
        for row, o in zip(rows, clipscore(cands, images)): row.update(o)
    json.dump(rows, open(os.path.join(RES, out_name), "w"), indent=1); print("wrote", out_name, len(rows))

if __name__ == "__main__":
    what = sys.argv[1]
    data = json.load(open(os.path.join(HERE, "golden_dataset_ext.json" if what == "ext" else "golden_dataset.json")))
    byid = {it["id"]: it for it in data}; rows, cands, refs, images = [], [], [], []
    if what in ("contrast", "ext"):
        for it in data:
            for k in "ABC":
                rows.append(dict(item=it["id"], category=it["category"], cand=k)); cands.append(it["candidates"][k]); refs.append(it["reference"]); images.append(it.get("image"))
        if what == "ext":   # the held-out set also needs the base metrics of Experiment 1
            from metrics import overlap, f_syntax
            for row, o in zip(rows, overlap(cands, refs)): row.update(o)
            for row, c in zip(rows, cands): row["f_syntax"] = f_syntax(c)
        run(rows, cands, refs, images, "ext_metrics_contrast.json" if what == "contrast" else "extset_metrics_ext.json")
    else:
        files = ["gen_qwen3vl-2b.jsonl", "gen_smolvlm2-2.2b.jsonl"] if what == "gen" else ["gen_star_qwen3vl-2b.jsonl", "gen_star_smolvlm2-2.2b.jsonl"]
        for f in files:
            for l in open(os.path.join(RES, f)):
                g = json.loads(l); rows.append(dict(item=g["item"], category=g["category"], generator=g["generator"]))
                cands.append(strip_markdown(g["output"]) if g["output"].strip() else "(empty)"); refs.append(byid[g["item"]]["reference"]); images.append(byid[g["item"]].get("image"))
        run(rows, cands, refs, images, f"ext_metrics_{what}.json")
