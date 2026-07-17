#!/usr/bin/env python3
"""Same-seed comparison, STACKED pose-error composition: marching-only vs both across 3 seeds
(0.1+0.3 mix, SNR-0.3 particles). Shows loss-ranking stabilizes convergence. Arial + SVG."""
import os, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42; plt.rcParams['axes.linewidth']=0.8; plt.rcParams['savefig.bbox']='tight'
NPZ='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/results/2026_Jul_06/ablation_npz'
OUT='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/mix_figures/'
BINS=[0,5,10,30,90,180]; LAB=['<5°','5–10°','10–30°','30–90°','>90°']; BC=['#1a7d1a','#7bc043','#f6d743','#f0932b','#c0392b']
def comp(name):
    r=np.load(os.path.join(NPZ,name+'.npz'))['res']
    return [100*np.mean((r>=BINS[i])&(r<BINS[i+1])) for i in range(len(BINS)-1)]
seeds=[0,1,2]
march=['mix_lrOFF_hi','mix_marchonly_seed1_hi','mix_marchonly_seed2_hi']
both =['mix_lrON_hi','mix_both_seed1_hi','mix_both_seed2_hi']
w=0.34
fig,ax=plt.subplots(figsize=(8.0,4.7))
for si,s in enumerate(seeds):
    for arm_npz,off,tag in [(march[si],-w/2-0.01,'march'),(both[si],+w/2+0.01,'both')]:
        C=comp(arm_npz); bot=0
        for j in range(len(LAB)):
            ax.bar(si+off,C[j],bottom=bot,width=w,color=BC[j],edgecolor='white',lw=0.6,zorder=3); bot+=C[j]
        g=C[0]
        if g>=8: ax.text(si+off,g/2,f'{g:.0f}%',ha='center',va='center',color='white',fontsize=8.5,fontweight='bold')
        else:    ax.text(si+off,g+2,f'{g:.0f}%',ha='center',va='bottom',color='#222',fontsize=8)
        ax.text(si+off,-4,tag,ha='center',va='top',fontsize=8,color=('#1f7a2a' if tag=='march' else '#2f6db3'))
ax.set_xticks(seeds); ax.set_xticklabels([f'seed {s}' for s in seeds],fontsize=10.5); ax.tick_params(axis='x',pad=18)
ax.set_ylabel('particles (%)',fontsize=11.5); ax.set_ylim(0,100); ax.set_xlim(-0.6,len(seeds)-0.4)
ax.set_title('Loss-ranking stabilizes ab-initio convergence — 0.1+0.3 mix, SNR-0.3 particles\n'
             '(same seed per pair;  marching-only converges 1/3 seeds,  +loss-rank 3/3)',fontsize=10.5)
ax.legend(handles=[Patch(fc=BC[j],label=LAB[j]) for j in range(len(LAB))],frameon=False,fontsize=9,ncol=5,
          loc='lower center',bbox_to_anchor=(0.5,-0.20))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(labelsize=9.5)
plt.tight_layout()
for ext in ('svg','pdf','png'): fig.savefig(OUT+'same_seed_rescue.'+ext,dpi=200)
print('saved same_seed_rescue.{svg,pdf,png} (stacked)')
