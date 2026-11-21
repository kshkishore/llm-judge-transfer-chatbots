import os, csv, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ['texgyretermes-regular.otf','texgyretermes-bold.otf','texgyretermes-italic.otf']:
    fm.fontManager.addfont(os.path.join(os.environ.get('TERMES_DIR','.'),f))
plt.rcParams.update({'font.family':'TeX Gyre Termes','font.size':10,'axes.linewidth':0.6})
rows=[r for r in csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','gen_table_v2.csv'))) if r['J2'] not in ('','nan','None')]
rng=np.random.default_rng(1)
fig,axes=plt.subplots(1,2,figsize=(6.6,3.1),sharey=True)
cols={'qwen3vl-2b':'#5b7fb5','smolvlm2-2.2b':'#c9a227'}
names={'qwen3vl-2b':'G1 (Qwen3-VL-2B)','smolvlm2-2.2b':'G2 (SmolVLM2-2.2B)'}
for ax,g in zip(axes,['qwen3vl-2b','smolvlm2-2.2b']):
    rs=[r for r in rows if r['generator']==g]
    x=np.array([float(r['J1']) for r in rs]); y=np.array([float(r['J2']) for r in rs])
    ok=np.array([r['fmt_ok']=='True' for r in rs])
    jx=rng.uniform(-0.012,0.012,len(x)); jy=rng.uniform(-0.012,0.012,len(y))
    ax.plot([0,1],[0,1],color='#bbbbbb',lw=0.7,ls='--',zorder=0)
    ax.scatter(x[ok]+jx[ok],y[ok]+jy[ok],s=16,facecolor=cols[g],edgecolor='white',linewidth=0.4,label='meets format rule',zorder=3)
    ax.scatter(x[~ok]+jx[~ok],y[~ok]+jy[~ok],s=16,facecolor='white',edgecolor=cols[g],linewidth=0.8,label='breaks format rule',zorder=3)
    ax.set_title(names[g],fontsize=10)
    ax.set_xlim(-0.03,1.03); ax.set_ylim(-0.03,1.05)
    ax.set_xlabel('J1 (Qwen3-VL-4B) score')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.tick_params(width=0.6,length=2.5)
    ax.legend(loc='lower right',fontsize=8,frameon=False,handletextpad=0.2)
axes[0].set_ylabel('J2 (Phi-4-mini) score')
fig.tight_layout(pad=0.4)
fig.savefig('fig1.png',dpi=600)
