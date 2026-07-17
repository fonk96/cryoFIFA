#!/usr/bin/env python3
"""Loss-ranked selection: uniform vs mixed SNR. Panel A = reversal, Panel B = loss separation."""
import os, glob, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Arial'
plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42
plt.rcParams['axes.linewidth']=0.8

OUT='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/mix_figures'
os.makedirs(OUT,exist_ok=True)
R='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/results/2026_Jul_06'

# ---- Panel A data (validated pose_bestof8, % solved) ----
regimes=['Uniform\nSNR 0.3','Mix\n0.1 + 0.3','Mix\n0.1 + 0.5']
# [<5deg, <10deg] per arm
march = {'lt5':[80.7,0.0,4.0], 'lt10':[81.0,0.1,10.8]}
lrank = {'lt5':[69.5,3.6,10.0],'lt10':[70.0,13.0,15.6]}

# ---- Panel B: per-particle loss split by SNR source (mix0105 lrON ep70) ----
HIGH='J532'
loss_hi=[]; loss_lo=[]
for fn in glob.glob(os.path.join(R,'mix0105_lrON_loss_ep70','loss_infer_rank*.txt')):
    for ln in open(fn):
        t=ln.split()
        if len(t)<2: continue
        try: v=float(t[-1])
        except: continue
        (loss_hi if HIGH in t[0] else loss_lo).append(v)
loss_hi=np.array(loss_hi); loss_lo=np.array(loss_lo)
med=np.median(np.concatenate([loss_hi,loss_lo]))   # 50% keep threshold

fig,(axA,axB)=plt.subplots(1,2,figsize=(9.2,3.8))

# ===== Panel A: grouped bars, <5deg (dark) stacked under 5-10deg (light) =====
x=np.arange(len(regimes)); w=0.36
cM,cL='#9099a2','#2f6db3'
for i,(arm,off,c) in enumerate([(march,-w/2,cM),(lrank,+w/2,cL)]):
    lt5=np.array(arm['lt5']); lt10=np.array(arm['lt10'])
    axA.bar(x+off,lt5,w,color=c,edgecolor='k',linewidth=0.6,zorder=3)
    axA.bar(x+off,lt10-lt5,w,bottom=lt5,color=c,alpha=0.4,edgecolor='k',linewidth=0.6,zorder=3)
    for xi,v in zip(x+off,lt10):
        axA.text(xi,v+1.2,f'{v:.0f}',ha='center',va='bottom',fontsize=8)
axA.set_xticks(x); axA.set_xticklabels(regimes,fontsize=9)
axA.set_ylabel('particles solved (%)',fontsize=10)
axA.set_ylim(0,92)
axA.set_title('Loss-ranking reverses sign with data composition',fontsize=10)
# legend
from matplotlib.patches import Patch
axA.legend(handles=[Patch(fc=cM,ec='k',label='march only'),
                    Patch(fc=cL,ec='k',label='+ loss-rank'),
                    Patch(fc='0.3',alpha=1,label='<5°'),Patch(fc='0.3',alpha=0.4,label='5–10°')],
           fontsize=7.5,frameon=False,loc='upper right',ncol=2,columnspacing=1.0,handlelength=1.2)
# annotate direction
axA.annotate('loss-rank\nhurts',xy=(0,74),ha='center',fontsize=7.5,color=cL)
axA.annotate('loss-rank helps',xy=(1.5,32),ha='center',fontsize=7.5,color=cL)

# ===== Panel B: loss distributions high vs low SNR =====
lo,hi=1000,20000
bins=np.linspace(lo,hi,60)
axB.hist(loss_hi,bins=bins,color=cL,alpha=0.8,label=f'high-SNR (0.5, n={len(loss_hi)//1000}k)',zorder=3)
axB.hist(loss_lo,bins=bins,color='#c44',alpha=0.65,label=f'low-SNR (0.1, n={len(loss_lo)//1000}k)',zorder=2)
axB.axvline(med,color='k',ls='--',lw=1.0,zorder=4)
ymax=axB.get_ylim()[1]
axB.axvspan(lo,med,color=cL,alpha=0.05,zorder=0)
axB.axvspan(med,hi,color='#c44',alpha=0.05,zorder=0)
axB.text(med-350,ymax*0.60,'kept 50%\n(→ clean half)',fontsize=8,va='center',ha='right',color=cL)
axB.text(med+350,ymax*0.60,'rejected 50%\n(junk)',fontsize=8,va='center',ha='left',color='#a33')
axB.set_xlabel('per-particle reconstruction loss (ep70)',fontsize=10)
axB.set_ylabel('particles',fontsize=10)
axB.set_title('Loss cleanly separates the SNR populations',fontsize=10)
axB.set_xlim(lo,hi)
axB.legend(fontsize=8,frameon=False,loc='upper right')

for ax in (axA,axB):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8.5)
plt.tight_layout()
for ext in ('png','svg','pdf'):
    fig.savefig(os.path.join(OUT,f'mix_lossrank_reversal.{ext}'),dpi=200,bbox_inches='tight')
print('saved',OUT,'mix_lossrank_reversal.{png,svg,pdf}')
print(f'loss sep: high median={np.median(loss_hi):.0f}  low median={np.median(loss_lo):.0f}  50%thr={med:.0f}')
