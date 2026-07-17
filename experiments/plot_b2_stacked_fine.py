#!/usr/bin/env python3
"""B2 half1 ablation, FINE stacked pose-error composition (vs cryoSPARC): splits the sub-5 deg
"solved" zone into <1 / 1-3 / 3-5 deg to test whether the local-refine module wins at high pose
precision. It does NOT: local-refine is uniformly worse (fewer <1 and <3 particles too).
Arial + SVG. norefine/localrefine from scratchpad, coarseonly(+32) from cluster ablation_npz."""
import os, re, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42; plt.rcParams['axes.linewidth']=0.8; plt.rcParams['savefig.bbox']='tight'
SP='/tmp/claude-1000/-sddn-kongfang-work-cryoFIFA-V260706-cryoFIFA/5b6a4386-0d4c-46ae-a59a-407ec3a82e9b/scratchpad/'
NPZ='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/results/2026_Jul_06/ablation_npz/'
LOG='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/LOG/'
OUT='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/lrab_figures/'; os.makedirs(OUT,exist_ok=True)
# FINE bins: split the solved zone (<5 deg) into <1 / 1-3 / 3-5.
BINS=[0,1,3,5,10,30,90,180]
LAB=['<1°','1–3°','3–5°','5–10°','10–30°','30–90°','>90°']
BC =['#0d5f0d','#2e8b2e','#7bc043','#f6d743','#f0932b','#d35400','#c0392b']  # 3 greens = solved zone
def comp(r): return [100*np.mean((r>=BINS[i])&(r<BINS[i+1])) for i in range(len(BINS)-1)]
def load(*paths):
    p=next((x for x in paths if os.path.exists(x)),None)
    return np.load(p)['res'] if p else None
tmin={}
if os.path.exists(LOG+'ablation_times.log'):
    for line in open(LOG+'ablation_times.log'):
        m=re.match(r'TRAIN_SECONDS\s+(\w+)\s+([\d.]+)',line.strip())
        if m: tmin[m.group(1)]=float(m.group(2))/60.0
ARMS=[('no local-refine\n(single-res)', load(SP+'ablation_norefine.npz',NPZ+'ablation_norefine.npz'),           tmin.get('norefine')),
      ('+ local-refine',                 load(SP+'ablation_localrefine.npz',NPZ+'ablation_localrefine.npz'),     tmin.get('localrefine')),
      ('coarse-only\n(r48, 96px)',       load(NPZ+'ablation_coarseonly.npz'),                                    tmin.get('coarseonly')),
      ('coarse-only\n(r32, 64px)',       load(NPZ+'ablation_coarseonly32.npz'),                                  tmin.get('coarseonly32'))]
ARMS=[a for a in ARMS if a[1] is not None]
fig,ax=plt.subplots(figsize=(1.7*len(ARMS)+1.6,5.2))
x=np.arange(len(ARMS))
for xi,(lab,r,t) in zip(x,ARMS):
    C=comp(r); bot=0
    for j in range(len(LAB)):
        ax.bar(xi,C[j],bottom=bot,width=0.6,color=BC[j],edgecolor='white',lw=0.6,zorder=3)
        # label the three solved-zone segments (greens) if tall enough
        if j<3 and C[j]>=2.2:
            ax.text(xi,bot+C[j]/2,f'{C[j]:.1f}',ha='center',va='center',
                    color='white',fontsize=8.5,fontweight='bold')
        bot+=C[j]
    # cumulative solved markers on TOP of each bar (single line, clear of title)
    c1,c3,c5=100*np.mean(r<1),100*np.mean(r<3),100*np.mean(r<5)
    ax.text(xi,101,f'<1°:{c1:.1f}  <3°:{c3:.1f}  <5°:{c5:.0f}%',ha='center',va='bottom',
            fontsize=8.5,color='#222')
    if t: ax.text(xi,-5.5,f'{t:.0f} min',ha='center',va='top',fontsize=10,fontweight='bold',color='#333')
ax.set_xticks(x); ax.set_xticklabels([a[0] for a in ARMS],fontsize=9.5); ax.tick_params(axis='x',pad=24)
ax.set_ylabel('particles (%)',fontsize=11.5); ax.set_ylim(0,100); ax.set_xlim(-0.65,len(ARMS)-0.35)
ax.set_title('B2 half1: FINE pose-error composition vs cryoSPARC\n(solved zone split <1° / 1–3° / 3–5° — local-refine does NOT win at high precision)',fontsize=10,pad=26)
ax.legend(handles=[Patch(fc=BC[j],label=LAB[j]) for j in range(len(LAB))],
          frameon=False,fontsize=8.5,ncol=7,loc='lower center',bbox_to_anchor=(0.5,-0.30),columnspacing=1.1,handletextpad=0.5)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(labelsize=9.5)
plt.tight_layout()
for ext in ('svg','pdf','png'): fig.savefig(OUT+'b2_pose_stacked_fine.'+ext,dpi=200)
print('saved b2_pose_stacked_fine.{svg,pdf,png} ->',OUT)
print(f'{"arm":24s} {"<1":>6s} {"<3":>6s} {"<5":>6s} {"median":>7s} {"time":>7s}')
for lab,r,t in ARMS:
    print(f'{lab.replace(chr(10)," "):24s} {100*np.mean(r<1):5.1f}% {100*np.mean(r<3):5.1f}% {100*np.mean(r<5):5.1f}% {np.median(r):6.2f}° {("%.0fmin"%t) if t else "   -":>7s}')
