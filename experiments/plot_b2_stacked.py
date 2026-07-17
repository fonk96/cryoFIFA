#!/usr/bin/env python3
"""B2 half1 local-refine ablation, STACKED pose-error composition (vs cryoSPARC): norefine vs localrefine
vs coarse-only, with training time under each bar. Arial + SVG. Run after coarse-only completes."""
import os, re, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42; plt.rcParams['axes.linewidth']=0.8; plt.rcParams['savefig.bbox']='tight'
SP='/tmp/claude-1000/-sddn-kongfang-work-cryoFIFA-V260706-cryoFIFA/5b6a4386-0d4c-46ae-a59a-407ec3a82e9b/scratchpad/'
NPZ='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/results/2026_Jul_06/ablation_npz/'
LOG='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/LOG/'
OUT='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/lrab_figures/'; os.makedirs(OUT,exist_ok=True)
BINS=[0,5,10,30,90,180]; LAB=['<5°','5–10°','10–30°','30–90°','>90°']; BC=['#1a7d1a','#7bc043','#f6d743','#f0932b','#c0392b']
def comp(r): return [100*np.mean((r>=BINS[i])&(r<BINS[i+1])) for i in range(5)]
def load(p): return np.load(p)['res'] if os.path.exists(p) else None
# training minutes (from ablation_times.log; only the LAST completed run per arm)
tmin={}
for line in open(LOG+'ablation_times.log'):
    m=re.match(r'TRAIN_SECONDS\s+(\w+)\s+([\d.]+)',line.strip())
    if m: tmin[m.group(1)]=float(m.group(2))/60.0
ARMS=[('no local-refine\n(single-res)', load(SP+'ablation_norefine.npz'),      tmin.get('norefine')),
      ('+ local-refine',                 load(SP+'ablation_localrefine.npz'),   tmin.get('localrefine')),
      ('coarse-only\n(low-res encoder)', load(NPZ+'ablation_coarseonly.npz'),   tmin.get('coarseonly'))]
fig,ax=plt.subplots(figsize=(7.4,4.8))
x=np.arange(len(ARMS))
for xi,(lab,r,t) in zip(x,ARMS):
    if r is None:
        ax.text(xi,50,'pending',ha='center',va='center',color='#bbb',fontsize=11,rotation=90); continue
    C=comp(r); bot=0
    for j in range(5):
        ax.bar(xi,C[j],bottom=bot,width=0.62,color=BC[j],edgecolor='white',lw=0.6,zorder=3); bot+=C[j]
    ax.text(xi,C[0]/2,f'{C[0]:.0f}%',ha='center',va='center',color='white',fontsize=10,fontweight='bold')
    if t: ax.text(xi,-6,f'{t:.0f} min',ha='center',va='top',fontsize=9.5,fontweight='bold',color='#333')
ax.set_xticks(x); ax.set_xticklabels([a[0] for a in ARMS],fontsize=9.5); ax.tick_params(axis='x',pad=22)
ax.set_ylabel('particles (%)',fontsize=11.5); ax.set_ylim(0,100); ax.set_xlim(-0.6,len(ARMS)-0.4)
ax.set_title('B2 half1: pose-error composition vs cryoSPARC + training time\n(local-refine module = training-time optimization, accuracy held)',fontsize=10.5)
ax.legend(handles=[Patch(fc=BC[j],label=LAB[j]) for j in range(5)],frameon=False,fontsize=9,ncol=5,loc='lower center',bbox_to_anchor=(0.5,-0.37))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(labelsize=9.5)
plt.tight_layout()
for ext in ('svg','pdf','png'): fig.savefig(OUT+'b2_pose_stacked.'+ext,dpi=200)
print('saved b2_pose_stacked.{svg,pdf,png}')
for lab,r,t in ARMS:
    if r is not None: print(f'  {lab.splitlines()[0]:20s} <5={100*np.mean(r<5):.1f}%  time={t:.0f}min' if t else f'  {lab}: <5={100*np.mean(r<5):.1f}%')
