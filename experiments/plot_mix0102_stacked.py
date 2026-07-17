#!/usr/bin/env python3
"""mix 0.1+0.2 (harder mix) 4-arm ablation, STACKED pose-error composition, 2 panels (SNR0.1 | SNR0.2).
Shows base/march-only/LR-only all collapse; only both (march+loss-rank) rescues -> synergy. Arial+SVG."""
import os, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'; plt.rcParams['pdf.fonttype']=42
plt.rcParams['axes.linewidth']=0.8; plt.rcParams['savefig.bbox']='tight'
NPZ='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/results/2026_Jul_06/ablation_npz/'
OUT='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/mix_figures/'; os.makedirs(OUT,exist_ok=True)
BINS=[0,5,10,30,90,180]; LAB=['<5°','5–10°','10–30°','30–90°','>90°']; BC=['#1a7d1a','#7bc043','#f6d743','#f0932b','#c0392b']
ARMS=[('neither','base'),('lrOFF','marching\nonly'),('lossrankonly','loss-rank\nonly'),('lrON','both\n(march+LR)')]
def comp(r): return [100*np.mean((r>=BINS[i])&(r<BINS[i+1])) for i in range(5)]
def load(a,sub):
    p=f'{NPZ}mix0102_{a}_{sub}.npz'; return np.load(p)['res'] if os.path.exists(p) else None
fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
for ax,(sub,title) in zip(axes,[('lo','SNR 0.1 subpopulation'),('hi','SNR 0.2 subpopulation')]):
    x=np.arange(len(ARMS))
    for xi,(arm,albl) in zip(x,ARMS):
        r=load(arm,sub)
        if r is None: ax.text(xi,50,'--',ha='center'); continue
        C=comp(r); bot=0
        for j in range(5):
            ax.bar(xi,C[j],bottom=bot,width=0.66,color=BC[j],edgecolor='white',lw=0.6,zorder=3); bot+=C[j]
        ax.text(xi,C[0]+1.5,f'{C[0]:.0f}%',ha='center',va='bottom',fontsize=10,fontweight='bold',
                color='#1a7d1a' if C[0]>3 else '#999')
    ax.set_xticks(x); ax.set_xticklabels([a[1] for a in ARMS],fontsize=9.5)
    ax.set_title(title,fontsize=11.5); ax.set_ylim(0,100); ax.set_xlim(-0.6,len(ARMS)-0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(labelsize=9.5)
axes[0].set_ylabel('particles (%)',fontsize=11.5)
fig.legend(handles=[Patch(fc=BC[j],label=LAB[j]) for j in range(5)],frameon=False,fontsize=9.5,
           ncol=5,loc='lower center',bbox_to_anchor=(0.5,-0.02))
fig.suptitle('cryoFIFA on the 0.1+0.2 mix: only march + loss-rank (both) rescues the SNR0.2 subpopulation',fontsize=12,y=1.0)
plt.tight_layout(rect=[0,0.05,1,1])
for ext in ('svg','pdf','png'): fig.savefig(OUT+'mix0102_stacked.'+ext,dpi=200)
print('saved mix0102_stacked.{svg,pdf,png} ->',OUT)
for a,_ in ARMS:
    for sub,nm in [('lo','SNR0.1'),('hi','SNR0.2')]:
        r=load(a,sub)
        if r is not None: print(f'  {a:14s} {nm}: <5={100*np.mean(r<5):.1f}%  median={np.median(r):.1f}')
