#!/usr/bin/env python3
"""Pose-error comparison for the 3x4 ablation: one plot per condition, 4 arms overlaid (<=4 histograms).
Step histograms so the 4 arms are directly comparable. Arial + SVG."""
import os, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42; plt.rcParams['axes.linewidth']=0.8; plt.rcParams['savefig.bbox']='tight'
BASE='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments'
NPZ=os.path.join(BASE,'results/2026_Jul_06/ablation_npz'); OUT=os.path.join(BASE,'mix_figures'); os.makedirs(OUT,exist_ok=True)
BINS=np.linspace(0,180,37)
# arm -> (label, color). order = best-to-worst-ish for legend readability
ARMS=[('both','march + loss-rank','#2f6db3'),('marchonly','march only','#2a9d3a'),
      ('lossrankonly','loss-rank only','#e08214'),('neither','neither','#8a8a8a')]

def load(name):
    p=os.path.join(NPZ,name+'.npz'); return np.load(p)['res'] if os.path.exists(p) else None

def panel(fname, title, resolver):
    fig,ax=plt.subplots(figsize=(6.2,4.2))
    ax.axvspan(0,5,color='#3a3',alpha=0.07,zorder=0)
    ymax=0
    for arm,lab,c in ARMS:
        r=resolver(arm)
        if r is None: continue
        h,_=np.histogram(r,bins=BINS)
        ymax=max(ymax,h.max())
        ax.hist(r,bins=BINS,histtype='step',lw=2.0,color=c,label=f'{lab}  ({100*np.mean(r<5):.0f}% <5°)',zorder=3)
    ax.set_yscale('log'); ax.set_ylim(20,ymax*1.6)
    ax.set_xlabel('per-particle pose error (°)',fontsize=11); ax.set_ylabel('particles (log)',fontsize=11)
    ax.set_xlim(0,180); ax.set_title(title,fontsize=11)
    ax.legend(fontsize=9,frameon=False,loc='upper right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(labelsize=9.5)
    plt.tight_layout()
    for ext in ('svg','pdf','png'): fig.savefig(os.path.join(OUT,fname+'.'+ext),dpi=200)
    plt.close(fig); print('saved',fname)

MIX={'both':'mix_lrON','marchonly':'mix_lrOFF','lossrankonly':'mix_lossrankonly','neither':'mix_neither'}
panel('histcmp_snr01','Pose error by arm — SNR 0.1 (floor)', lambda a: load('snr01_'+a))
panel('histcmp_snr03','Pose error by arm — SNR 0.3 (clean)', lambda a: load('snr03_'+a))
panel('histcmp_mix_high','Pose error by arm — 0.1+0.3 mix, high-SNR (0.3) particles', lambda a: load(MIX[a]+'_hi'))
panel('histcmp_mix_low','Pose error by arm — 0.1+0.3 mix, low-SNR (0.1) particles', lambda a: load(MIX[a]+'_lo'))
