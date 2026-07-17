#!/usr/bin/env python3
"""Stacked pose-error composition for the 3x4 ablation: 3 partitions (regimes), 4 stacked bars each
(base / marching only / loss-rank only / both), split into error bins. Arial + SVG."""
import os, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42; plt.rcParams['axes.linewidth']=0.8; plt.rcParams['savefig.bbox']='tight'
BASE='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments'
NPZ=os.path.join(BASE,'results/2026_Jul_06/ablation_npz'); OUT=os.path.join(BASE,'mix_figures/'); os.makedirs(OUT,exist_ok=True)
BINS=[0,5,10,30,90,180]; LAB=['<5°','5–10°','10–30°','30–90°','>90°']
BC=['#1a7d1a','#7bc043','#f6d743','#f0932b','#c0392b']
CFG=[('base','neither'),('marching\nonly','marchonly'),('loss-rank\nonly','lossrankonly'),('both','both')]
MIX={'neither':'mix_neither','marchonly':'mix_lrOFF','lossrankonly':'mix_lossrankonly','both':'mix_lrON'}  # all seed 0 (same-seed: shows loss-rank rescuing marchonly's seed-0 collapse)

def load(name):
    p=os.path.join(NPZ,name+'.npz'); return np.load(p)['res'] if os.path.exists(p) else None
def comp(r): return [100*np.mean((r>=BINS[i])&(r<BINS[i+1])) for i in range(len(BINS)-1)] if r is not None else [0]*(len(BINS)-1)
REG=[('SNR 0.1  (below threshold)', lambda k: load('snr01_'+k)),
     ('SNR 0.3  (clean)',           lambda k: load('snr03_'+k)),
     ('0.1+0.3 mix  (0.3 particles)',lambda k: load(MIX[k]+'_hi'))]

fig,axes=plt.subplots(1,3,figsize=(12.6,4.5),sharey=True)
x=np.arange(len(CFG))
for ax,(title,res) in zip(axes,REG):
    C=np.array([comp(res(key)) for _,key in CFG])
    bot=np.zeros(len(CFG))
    for j in range(len(LAB)):
        ax.bar(x,C[:,j],bottom=bot,width=0.68,color=BC[j],edgecolor='white',lw=0.6,zorder=3)
        bot+=C[:,j]
    # solved (<5) percentage: inside the green segment when large enough, else just above the bar base
    for xi in x:
        g=C[xi,0]
        if g>=8: ax.text(xi,g/2,f'{g:.0f}%',ha='center',va='center',fontsize=8.5,fontweight='bold',color='white')
        else:    ax.text(xi,g+2,f'{g:.0f}%',ha='center',va='bottom',fontsize=8,color='#222')
    ax.set_xticks(x); ax.set_xticklabels([lab for lab,_ in CFG],fontsize=9.5)
    ax.set_title(title,fontsize=11); ax.set_ylim(0,100)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(labelsize=9.5); ax.margins(x=0.04)
axes[0].set_ylabel('particles (%)',fontsize=11.5)
fig.legend(handles=[Patch(fc=BC[j],label=LAB[j]) for j in range(len(LAB))],
           frameon=False,fontsize=9.5,ncol=5,loc='upper center',bbox_to_anchor=(0.5,1.05))
fig.suptitle('Pose-error composition by configuration across the three regimes   (label in bar = % solved <5°)',
             fontsize=11.5,y=1.10)
plt.tight_layout()
for ext in ('svg','pdf','png'): fig.savefig(OUT+'stacked_ablation.'+ext,dpi=200)
print('saved stacked_ablation.{svg,pdf,png}')
