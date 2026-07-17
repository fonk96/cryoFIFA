#!/usr/bin/env python3
"""Publication ablation: 3 datasets x 2x2 (marching x loss-rank). % solved <5deg.
Pending cells (still running) drawn hatched. Mix uses the high-SNR (0.3) subpopulation."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42; plt.rcParams['axes.linewidth']=0.8
OUT='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/mix_figures'; os.makedirs(OUT,exist_ok=True)

# %<5deg solved; None = still running.  arms keyed (marching, lossrank)
# march OFF: neither(LRoff), lossrankonly(LRon) ; march ON: marchonly(LRoff), both(LRon)
DAT={
 'SNR 0.1\n(below threshold)':      {'neither':0.0,'lossrankonly':0.0,'marchonly':0.0,'both':0.0},
 'SNR 0.3\n(clean, solvable)':      {'neither':50.1,'lossrankonly':39.0,'marchonly':80.7,'both':69.5},
 'SNR 0.1+0.3 mix\n(high-SNR half)':{'neither':38.2,'lossrankonly':43.1,'marchonly':0.0,'both':83.5},
}
cLRoff,cLRon='#9099a2','#2f6db3'
fig,axes=plt.subplots(1,3,figsize=(10.5,3.8),sharey=True)
for ax,(title,d) in zip(axes,DAT.items()):
    # x=0 marching OFF (neither, lossrankonly); x=1 marching ON (marchonly, both)
    specs=[(0,-0.2,'neither',cLRoff),(0,0.2,'lossrankonly',cLRon),
           (1,-0.2,'marchonly',cLRoff),(1,0.2,'both',cLRon)]
    for xc,off,arm,c in specs:
        v=d[arm]; xp=xc+off
        if v is None:
            ax.bar(xp,100,0.36,facecolor='none',edgecolor='#bbb',hatch='///',linewidth=0.6,zorder=2)
            ax.text(xp,4,'pending',rotation=90,ha='center',va='bottom',fontsize=7,color='#999')
        else:
            ax.bar(xp,v,0.36,color=c,edgecolor='k',linewidth=0.6,zorder=3)
            ax.text(xp,v+1.5,f'{v:.0f}',ha='center',va='bottom',fontsize=8.5,fontweight='bold')
    ax.set_xticks([0,1]); ax.set_xticklabels(['marching\nOFF','marching\nON'],fontsize=9)
    ax.set_title(title,fontsize=9.5)
    ax.set_ylim(0,100); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel('particles solved,  <5°  (%)',fontsize=10)
axes[1].legend(handles=[Patch(fc=cLRoff,ec='k',label='loss-rank OFF'),
                        Patch(fc=cLRon,ec='k',label='loss-rank ON')],
               fontsize=8.5,frameon=False,loc='upper center',ncol=1)
fig.suptitle('Frequency-marching × loss-ranking ablation  (ribosome, box 128)',fontsize=11,y=1.02)
plt.tight_layout()
for ext in ('png','svg','pdf'):
    fig.savefig(os.path.join(OUT,f'ablation_grid.{ext}'),dpi=200,bbox_inches='tight')
print('saved ablation_grid.{png,svg,pdf}')
