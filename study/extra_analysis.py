"""Additional analyses of the chatbot study, computed from logged outputs (gen_table_v2.csv, genjudge_*.jsonl)."""
import json,csv,re,statistics as st
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon
import os
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'results')+'/'
rows=list(csv.DictReader(open(R+'gen_table_v2.csv')))
def f(x):
    try: return float(x)
    except: return None
out={}
gens=['qwen3vl-2b','smolvlm2-2.2b']
# A: dimension means per judge per generator
dims='C O S T A R'.split()
A={}
for g in gens:
    rs=[r for r in rows if r['generator']==g]
    A[g]={j:{d:round(st.mean([f(r[f'{j}_{d}']) for r in rs if f(r[f'{j}_{d}']) is not None]),2) for d in dims} for j in ['J1','J2']}
out['dim_means']=A
# B: J1-J2 mean difference per dimension on text replies
txt=[r for r in rows if f(r['J2']) is not None]
out['n_text']=len(txt)
out['J2_minus_J1']={d:round(st.mean([f(r['J2_'+d])-f(r['J1_'+d]) for r in txt]),2) for d in dims}
# C: within-generator correlation of J1 with length
out['rho_len_within']={g:round(spearmanr([f(r['n_words']) for r in rows if r['generator']==g],[f(r['J1']) for r in rows if r['generator']==g])[0],3) for g in gens}
out['rho_len_within_J2']={g:round(spearmanr([f(r['n_words']) for r in txt if r['generator']==g],[f(r['J2']) for r in txt if r['generator']==g])[0],3) for g in gens}
# D: J1 score by format compliance within generator
D={}
for g in gens:
    ok=[f(r['J1']) for r in rows if r['generator']==g and r['fmt_ok'] in ('True','1','true')]
    no=[f(r['J1']) for r in rows if r['generator']==g and r['fmt_ok'] in ('False','0','false')]
    D[g]={'n_ok':len(ok),'n_fail':len(no),'mean_ok':round(st.mean(ok),3),'mean_fail':round(st.mean(no),3),'p':float('%.3g'%mannwhitneyu(ok,no).pvalue)}
out['J1_by_format']=D
D2={}
for g in gens:
    ok=[f(r['J2']) for r in txt if r['generator']==g and r['fmt_ok'] in ('True','1','true')]
    no=[f(r['J2']) for r in txt if r['generator']==g and r['fmt_ok'] in ('False','0','false')]
    D2[g]={'n_ok':len(ok),'n_fail':len(no),'mean_ok':round(st.mean(ok),3),'mean_fail':round(st.mean(no),3),'p':float('%.3g'%mannwhitneyu(ok,no).pvalue)}
out['J2_by_format']=D2
# E: diagnosis content
J1=[json.loads(l) for l in open(R+'genjudge_j1.jsonl')]; J2=[json.loads(l) for l in open(R+'genjudge_j2.jsonl')]
pat_form=re.compile(r'\b(verbose|long|length|lengthy|exceed\w*|sentence\w*|format\w*|concise|wordy|brief|too many|list)\b',re.I)
pat_mean=re.compile(r'\b(misinterpret\w*|misread\w*|misunderst\w*|literal\w*|incorrect\w*|wrong|fails? to (recognise|recognize|understand|address)|does not (recognise|recognize|address|understand))\b',re.I)
E={}
for name,L in [('J1',J1),('J2',J2)]:
    for g in gens:
        ds=[x['scores']['diagnosis'] for x in L if x['generator']==g]
        E[f'{name}|{g}']={'n':len(ds),'form':sum(bool(pat_form.search(d)) for d in ds),'meaning':sum(bool(pat_mean.search(d)) for d in ds)}
out['diagnosis_terms']=E
# F: disagreement by category (J1 flags misreading O<=2, J2 does not)
cat={}
for r in txt:
    j1=f(r['J1_O'])<=2; j2=f(r['J2_O'])<=2
    c=r['category']; cat.setdefault(c,[0,0,0,0])
    cat[c][0]+=j1 and j2; cat[c][1]+=j1 and not j2; cat[c][2]+=j2 and not j1; cat[c][3]+=1
out['misread_by_cat']=cat
# G: perfect profiles by judge, generator
out['all5']={j:{g:sum(1 for r in txt if r['generator']==g and all(f(r[f'{j}_{d}'])==5 for d in dims)) for g in gens} for j in ['J1','J2']}
# H: absolute differences: mean |J1-J2| on fprag
out['mean_abs_diff']=round(st.mean([abs(f(r['J1'])-f(r['J2'])) for r in txt]),3)
out['mean_J1_text']=round(st.mean([f(r['J1']) for r in txt]),3); out['mean_J2_text']=round(st.mean([f(r['J2']) for r in txt]),3)
# I: share of replies by generator where J1 and J2 disagree on preferred generator: already in summary
json.dump(out,open(R+'extra_results.json','w'),indent=1)
print(json.dumps(out,indent=1))
