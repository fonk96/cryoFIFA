#!/usr/bin/env python3
"""Per-particle pose-error histograms, 2x2 arm grid per SNR condition.
SNR 0.1 / SNR 0.3: one histogram per arm. Mix 0.1+0.3: high(0.3) vs low(0.1) overlaid."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42; plt.rcParams['axes.linewidth']=0.8
BASE='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments'
NPZ=os.path.join(BASE,'results/2026_Jul_06/ablation_npz'); OUT=os.path.join(BASE,'mix_figures'); os.makedirs(OUT,exist_ok=True)
BINS=np.linspace(0,180,37)   # 5-degree bins; first bin = solved (<5)
cHI,cLO='#2f6db3','#c44'

def load(name):
    p=os.path.join(NPZ,name+'.npz')
    return np.load(p)['res'] if os.path.exists(p) else None

# 2x2 arm layout: rows = marching OFF / ON, cols = loss-rank OFF / ON
CELLS=[('neither','marching OFF  ·  loss-rank OFF'),('lossrankonly','marching OFF  ·  loss-rank ON'),
       ('marchonly','marching ON  ·  loss-rank OFF'),('both','marching ON  ·  loss-rank ON')]

def pct5(r): return 100*np.mean(r<5) if r is not None and len(r) else float('nan')

def make(cond, npz_prefix, title, mix=False):
    fig,axes=plt.subplots(2,2,figsize=(8.4,6.0),sharex=True,sharey=not mix)
    MIXMAP={'both':'lrON','marchonly':'lrOFF','lossrankonly':'lossrankonly','neither':'neither'}
    for ax,(arm,lab) in zip(axes.flat,CELLS):
        if mix:
            key=MIXMAP[arm]
            rh=load(f'mix_{key}_hi'); rl=load(f'mix_{key}_lo')
            if rh is None and rl is None:
                ax.text(0.5,0.5,'pending',transform=ax.transAxes,ha='center',va='center',fontsize=12,color='#bbb');
                ax.set_title(f'{arm}',fontsize=9)
            else:
                if rh is not None: ax.hist(rh,bins=BINS,color=cHI,alpha=0.75,label=f'SNR 0.3  ({pct5(rh):.0f}% <5°)')
                if rl is not None: ax.hist(rl,bins=BINS,color=cLO,alpha=0.6, label=f'SNR 0.1  ({pct5(rl):.0f}% <5°)')
                ax.legend(fontsize=7,frameon=False,loc='upper center')
                ax.set_title(f'{arm}   ({lab})',fontsize=8.5)
        else:
            r=load(f'{npz_prefix}_{arm}')
            if r is None:
                ax.text(0.5,0.5,'pending',transform=ax.transAxes,ha='center',va='center',fontsize=12,color='#bbb')
                ax.set_title(f'{arm}',fontsize=9)
            else:
                ax.hist(r,bins=BINS,color=cHI,alpha=0.8)
                ax.set_title(f'{arm}   ({lab})\n{pct5(r):.0f}% solved <5°',fontsize=8.5)
        ax.axvspan(0,5,color='#3a3',alpha=0.10,zorder=0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(labelsize=8)
    for ax in axes[-1]: ax.set_xlabel('per-particle pose error (°)',fontsize=9.5)
    for ax in axes[:,0]: ax.set_ylabel('particles',fontsize=9.5)
    fig.suptitle(title,fontsize=11,y=1.00)
    plt.tight_layout()
    for ext in ('png','svg','pdf'): fig.savefig(os.path.join(OUT,f'{cond}.{ext}'),dpi=200,bbox_inches='tight')
    print('saved',cond)

make('hist_snr01','snr01','Pose-error distribution — SNR 0.1  (below threshold)')
make('hist_snr03','snr03','Pose-error distribution — SNR 0.3  (clean, solvable)')
make('hist_mix0103','mix','Pose-error distribution — SNR 0.1+0.3 mix  (high 0.3 vs low 0.1)',mix=True)
