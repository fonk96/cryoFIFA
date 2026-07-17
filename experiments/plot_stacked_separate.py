#!/usr/bin/env python3
"""4 separate stacked pose-error composition plots (SNR0.1 / SNR0.3 / 0.1-in-mix / 0.3-in-mix),
each 4 configs (base/marching only/loss-rank only/both); plus an independent legend file. Arial + SVG."""
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
MIX={'neither':'mix_neither','marchonly':'mix_lrOFF','lossrankonly':'mix_lossrankonly','both':'mix_lrON'}
def load(name):
    p=os.path.join(NPZ,name+'.npz'); return np.load(p)['res'] if os.path.exists(p) else None
def comp(r): return [100*np.mean((r>=BINS[i])&(r<BINS[i+1])) for i in range(len(BINS)-1)] if r is not None else [0]*(len(BINS)-1)

COND=[('SNR 0.1',                         'stacked_snr01',    lambda k: load('snr01_'+k)),
      ('SNR 0.3',                         'stacked_snr03',    lambda k: load('snr03_'+k)),
      ('SNR 0.1 particles in 0.1+0.3 mix','stacked_mix_low',  lambda k: load(MIX[k]+'_lo')),
      ('SNR 0.3 particles in 0.1+0.3 mix','stacked_mix_high', lambda k: load(MIX[k]+'_hi'))]

x=np.arange(len(CFG))
for title,fname,res in COND:
    C=np.array([comp(res(key)) for _,key in CFG])
    fig,ax=plt.subplots(figsize=(4.0,4.2))
    bot=np.zeros(len(CFG))
    for j in range(len(LAB)):
        ax.bar(x,C[:,j],bottom=bot,width=0.72,color=BC[j],edgecolor='white',lw=0.6,zorder=3); bot+=C[:,j]
    for xi in x:
        g=C[xi,0]
        if g>=8: ax.text(xi,g/2,f'{g:.0f}%',ha='center',va='center',fontsize=8.5,fontweight='bold',color='white')
        else:    ax.text(xi,g+2,f'{g:.0f}%',ha='center',va='bottom',fontsize=8,color='#222')
    ax.set_xticks(x); ax.set_xticklabels([lab for lab,_ in CFG],fontsize=9.5)
    ax.set_ylabel('particles (%)',fontsize=11); ax.set_ylim(0,100); ax.set_title(title,fontsize=11)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(labelsize=9.5); ax.margins(x=0.05)
    plt.tight_layout()
    for ext in ('svg','pdf','png'): fig.savefig(OUT+fname+'.'+ext,dpi=200)
    plt.close(fig); print('saved',fname)

# independent legend file
figL,axL=plt.subplots(figsize=(4.6,0.6)); axL.axis('off')
axL.legend(handles=[Patch(fc=BC[j],label=LAB[j]) for j in range(len(LAB))],
           frameon=False,fontsize=10,ncol=5,loc='center',handlelength=1.2,columnspacing=1.1)
for ext in ('svg','pdf','png'): figL.savefig(OUT+'stacked_legend.'+ext,dpi=200)
print('saved stacked_legend')
