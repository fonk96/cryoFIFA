#!/usr/bin/env python3
"""How each module configuration performs across the three regimes. y = % particles solved (<5deg). Arial+SVG."""
import os, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42; plt.rcParams['axes.linewidth']=0.8; plt.rcParams['savefig.bbox']='tight'
OUT='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/mix_figures/'; os.makedirs(OUT,exist_ok=True)

arms=['march\n+ loss-rank','march\nonly','loss-rank\nonly','neither']
# %<5deg per regime (mix = the SNR-0.3 / high-SNR particles)
regimes=[('SNR 0.1  (below threshold)', [0,0,0,0],        '#b0b0b0'),
         ('SNR 0.3  (clean)',           [70,81,39,50],     '#2a9d3a'),
         ('0.1+0.3 mix  (0.3 particles)',[84,0,43,38],     '#2f6db3')]

x=np.arange(len(arms)); w=0.26
fig,ax=plt.subplots(figsize=(8.2,4.6))
for k,(lab,vals,c) in enumerate(regimes):
    off=(k-1)*w
    ax.bar(x+off,vals,w,color=c,edgecolor='k',linewidth=0.6,label=lab,zorder=3)
    for xi,v in zip(x+off,vals):
        ax.text(xi,v+1.3,f'{v:.0f}',ha='center',va='bottom',fontsize=8.5,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(arms,fontsize=10.5)
ax.set_ylabel('particles solved,  <5°  (%)',fontsize=11); ax.set_ylim(0,104)
ax.set_title('How each configuration performs across the three regimes',fontsize=12,pad=12)
ax.legend(fontsize=9.5,frameon=False,loc='upper right')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(labelsize=9.5)
plt.tight_layout()
for ext in ('svg','pdf','png'): fig.savefig(OUT+'solved_summary.'+ext,dpi=200)
print('saved solved_summary.{svg,pdf,png}')
