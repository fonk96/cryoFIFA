#!/usr/bin/env python3
"""Local-refine ablation figure for the reviewer response (ribosome sim, box 128, SNR 0.5).
Panel A: pose accuracy across arms/scales. Panel B: refiner correction magnitude vs epoch."""
import os, glob, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42; plt.rcParams['axes.linewidth']=0.8
BASE='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments'
R=os.path.join(BASE,'results/2026_Jul_06'); OUT=os.path.join(BASE,'lrab_figures'); os.makedirs(OUT,exist_ok=True)

# --- Panel A: pose accuracy (% <5deg) ---
labels=['standard\n(no module)','coarse-only\n(s=0)','local\ns=0.05','local\ns=0.1','local\ns=0.3']
pct=[94.0,91.2,91.1,92.2,0.1]
cols=['#555','#9099a2','#2f6db3','#2f6db3','#c0392b']

# --- Panel B: per-epoch median rotation correction, read refine_delta.txt ---
def traj(arm):
    ep={};
    for line in open(os.path.join(R,'lrab_ribo_%s'%arm,'refine_delta.txt')):
        t=line.split()
        if len(t)<2: continue
        e=int(t[0]); ep.setdefault(e,[]).append(float(t[1]))
    es=sorted(ep); return np.array(es), np.array([np.mean(ep[e]) for e in es])

fig,(axA,axB)=plt.subplots(1,2,figsize=(10,4.0),gridspec_kw={'width_ratios':[1.05,1]})

# Panel A
x=np.arange(len(labels))
axA.bar(x,pct,color=cols,edgecolor='k',linewidth=0.6,zorder=3)
for xi,v in zip(x,pct): axA.text(xi,v+1.5,f'{v:.0f}',ha='center',va='bottom',fontsize=9,fontweight='bold')
axA.axhspan(90,95,color='#3a3',alpha=0.06,zorder=0)
axA.set_xticks(x); axA.set_xticklabels(labels,fontsize=8)
axA.set_ylabel('particles solved,  <5°  (%)',fontsize=10); axA.set_ylim(0,102)
axA.set_title('Local-refine ablation (ribosome, box 128)',fontsize=10)
axA.annotate('coarse stage already\nsolves the poses',xy=(1,91),xytext=(1.7,55),fontsize=7.5,ha='center',
             arrowprops=dict(arrowstyle='->',lw=0.8,color='#555'))
axA.annotate('over-large scale\ndestabilizes',xy=(4,3),xytext=(3.4,40),fontsize=7.5,ha='center',color='#c0392b',
             arrowprops=dict(arrowstyle='->',lw=0.8,color='#c0392b'))
axA.spines['top'].set_visible(False); axA.spines['right'].set_visible(False); axA.tick_params(labelsize=8.5)

# Panel B
for arm,lab,c in [('scale0.05','s = 0.05  (stable, poses kept)','#2f6db3'),
                  ('scale0.1','s = 0.1',' #1f9e6b'.strip()),
                  ('coarselocal','s = 0.3  (poses collapse)','#c0392b')]:
    e,d=traj(arm); axB.plot(e,d,'-',color=c,lw=1.8,label=lab)
axB.axvline(32,color='0.6',ls=':',lw=1.0); axB.text(32.5,axB.get_ylim()[1]*0.02,'refiner engages\n(frontier > radius 32)',fontsize=7,va='bottom')
axB.set_xlabel('epoch',fontsize=10); axB.set_ylabel('median rotation correction  Δθ  (°)',fontsize=10)
axB.set_title('How much the refiner changes the pose',fontsize=10)
axB.legend(fontsize=8,frameon=False,loc='center right')
axB.set_xlim(30,70); axB.spines['top'].set_visible(False); axB.spines['right'].set_visible(False); axB.tick_params(labelsize=8.5)
plt.tight_layout()
for ext in ('png','svg','pdf'): fig.savefig(os.path.join(OUT,'localrefine_ablation.%s'%ext),dpi=200,bbox_inches='tight')
print('saved',OUT,'localrefine_ablation.{png,svg,pdf}')
