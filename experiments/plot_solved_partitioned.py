#!/usr/bin/env python3
"""One long figure, 3 partitions (regimes), 4 bars each (base/marching only/loss-rank only/both).
y = particles solved (<5deg, %). marchonly-on-mix = seed-0 collapse (same-seed as both -> shows the rescue). Arial+SVG."""
import os, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42; plt.rcParams['axes.linewidth']=0.8; plt.rcParams['savefig.bbox']='tight'
OUT='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/mix_figures/'; os.makedirs(OUT,exist_ok=True)

configs=['base','marching\nonly','loss-rank\nonly','both']
ccol=['#8a8a8a','#2a9d3a','#e08214','#2f6db3']
# %<5deg per regime, order [base, marching only, loss-rank only, both]
# mix marching-only = seed-1 convergent basin (78.6%)
regimes=[('SNR 0.1  (below threshold)', [0,0,0,0]),
         ('SNR 0.3  (clean)',           [50,81,39,70]),
         ('0.1+0.3 mix  (0.3 particles)',[38,0,43,84])]

x=np.arange(len(configs)); w=0.66
fig,axes=plt.subplots(1,3,figsize=(12.6,4.5),sharey=True)
for ax,(title,vals) in zip(axes,regimes):
    ax.bar(x,vals,width=w,color=ccol,edgecolor='k',linewidth=0.6,zorder=3)
    for xi,v in zip(x,vals): ax.text(xi,v+1.4,f'{v:.0f}',ha='center',va='bottom',fontsize=9.5,fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(configs,fontsize=9.5)
    ax.set_title(title,fontsize=11); ax.set_ylim(0,100)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(labelsize=9.5); ax.margins(x=0.06)
axes[0].set_ylabel('particles solved,  <5°  (%)',fontsize=11.5)
fig.suptitle('Pose accuracy by configuration across the three regimes',fontsize=12.5,y=1.02)
fig.text(0.5,-0.04,'all seed 0 (same seed): marching-only collapses (0%), loss-rank rescues the SAME basin (84%). marchonly is basin-variable (seeds: 0/0/79%)',
         ha='center',fontsize=8,color='#666')
plt.tight_layout()
for ext in ('svg','pdf','png'): fig.savefig(OUT+'solved_partitioned.'+ext,dpi=200)
print('saved solved_partitioned.{svg,pdf,png}')
