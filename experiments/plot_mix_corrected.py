#!/usr/bin/env python3
"""Corrected per-population pose accuracy: loss-rank rescues contaminated high-SNR.
Metric = median angular error (bars) + %<5deg annotated. Panel B = loss separation."""
import os, glob
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

# ---- corrected per-population numbers (strict stack-filtered pose_bestof8), 0.1+0.3 only ----
# baseline = SNR0.3 particles trained alone; then the SNR0.3 (high) and SNR0.1 (low) halves in the mix
groups=['SNR 0.3\nalone','SNR 0.3 particles\nin 0.1+0.3 mix','SNR 0.1 particles\nin 0.1+0.3 mix']
march_med=[1.40, 59.27, 110.05]   # median angular error, marchonly
lrank_med=[1.82,  1.43,  59.70]   #                        + loss-rank
march_pct=[80.7,  0.0,   0.0]     # %<5deg
lrank_pct=[69.5,  83.5,  23.5]

# ---- Panel B: loss separation (mix0103 lrON ep70) ----
HIGH='J530'
lh=[]; ll=[]
for fn in glob.glob(os.path.join(R,'mix0103_lrON_loss_ep70','loss_infer_rank*.txt')):
    for line in open(fn):
        t=line.split()
        if len(t)<2: continue
        try: v=float(t[-1])
        except: continue
        (lh if HIGH in t[0] else ll).append(v)
lh=np.array(lh); ll=np.array(ll); med=np.median(np.concatenate([lh,ll]))

fig,(axA,axB)=plt.subplots(1,2,figsize=(10.2,4.0),gridspec_kw={'width_ratios':[1.55,1]})
cM,cL='#9099a2','#2f6db3'
x=np.arange(len(groups)); w=0.38

# ===== Panel A: bar height = % solved (higher = better); median error as small gray sub-label =====
for arm_pct,arm_med,off,c,lab in [(march_pct,march_med,-w/2,cM,'march only'),
                                  (lrank_pct,lrank_med,+w/2,cL,'+ loss-rank')]:
    axA.bar(x+off,arm_pct,w,color=c,edgecolor='k',linewidth=0.6,zorder=3,label=lab)
    for xi,p,m in zip(x+off,arm_pct,arm_med):
        axA.text(xi,p+1.6,f'{p:.0f}%',ha='center',va='bottom',fontsize=8.5,fontweight='bold')
        mtxt=f'{m:.1f}°' if m<10 else f'{m:.0f}°'
        axA.text(xi,p+8.2,f'med {mtxt}',ha='center',va='bottom',fontsize=6.3,color='#888')
axA.set_xticks(x); axA.set_xticklabels(groups,fontsize=8.5)
axA.set_ylabel('particles solved,  <5°  (%)',fontsize=10)
axA.set_ylim(0,108)
axA.set_title('Loss-ranking rescues SNR 0.3 poses from SNR 0.1 contamination\n(20k SNR 0.1 + 20k SNR 0.3 mixture)',fontsize=9.5)
axA.legend(fontsize=8.5,frameon=False,loc='upper right')
# annotate the money shot: junk breaks marchonly (0%), loss-rank rescues (84%)
axA.annotate('0.1 junk breaks\nmarchonly (0%)',xy=(1-w/2,2),xytext=(0.40,40),fontsize=7.8,color='#333',
             ha='center',arrowprops=dict(arrowstyle='->',lw=0.8,color='#555'))
axA.annotate('loss-rank\nrescues (84%)',xy=(1+w/2,83.5),xytext=(1.62,58),fontsize=7.8,color=cL,
             ha='center',arrowprops=dict(arrowstyle='->',lw=0.9,color=cL))
axA.annotate('even 24% of\nthe 0.1 junk',xy=(2+w/2,23.5),xytext=(2.15,45),fontsize=7.3,color=cL,
             ha='center',arrowprops=dict(arrowstyle='->',lw=0.8,color=cL))

# ===== Panel B: loss separation =====
lo,hi=1000,20000; bins=np.linspace(lo,hi,60)
axB.hist(lh,bins=bins,color=cL,alpha=0.8,label=f'high-SNR 0.3 (n={len(lh)//1000}k)',zorder=3)
axB.hist(ll,bins=bins,color='#c44',alpha=0.65,label=f'low-SNR 0.1 (n={len(ll)//1000}k)',zorder=2)
axB.axvline(med,color='k',ls='--',lw=1.0,zorder=4)
ymax=axB.get_ylim()[1]
axB.axvspan(lo,med,color=cL,alpha=0.05); axB.axvspan(med,hi,color='#c44',alpha=0.05)
axB.text(med-350,ymax*0.6,'kept 50%\n(clean half)',fontsize=8,va='center',ha='right',color=cL)
axB.text(med+350,ymax*0.6,'rejected 50%\n(junk)',fontsize=8,va='center',ha='left',color='#a33')
axB.set_xlabel('per-particle loss (ep70)',fontsize=10); axB.set_ylabel('particles',fontsize=10)
axB.set_title('Loss separates the two SNR populations',fontsize=9.5)
axB.set_xlim(lo,hi); axB.legend(fontsize=8,frameon=False,loc='upper right')

for ax in (axA,axB):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)
plt.tight_layout()
for ext in ('png','svg','pdf'):
    fig.savefig(os.path.join(OUT,f'mix_lossrank_corrected.{ext}'),dpi=200,bbox_inches='tight')
print('saved mix_lossrank_corrected.{png,svg,pdf}')
print(f'panelB loss sep(0.1+0.3): high med={np.median(lh):.0f} low med={np.median(ll):.0f} thr={med:.0f}')
