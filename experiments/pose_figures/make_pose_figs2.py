import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import binned_statistic_2d, binned_statistic
rcParams.update({'font.size':12,'axes.linewidth':0.8,'font.family':'DejaVu Sans','savefig.bbox':'tight'})
SP='/tmp/claude-1000/-sddn-kongfang-work-cryoFIFA-V260706-cryoFIFA/5b6a4386-0d4c-46ae-a59a-407ec3a82e9b/scratchpad/'
h1=np.load(SP+'pose_h1.npz'); h2=np.load(SP+'pose_h2.npz')
def cat(k): return {'half1':h1[k],'half2':h2[k],'total':np.concatenate([h1[k],h2[k]])}
res=cat('res'); cfe=cat('cf_eul'); cse=cat('cs_eul'); copies=cat('copies')
C={'half1':'#0072B2','half2':'#D55E00','total':'#222222'}
RANGE=[[-180,180],[-90,90]]; BINS=64
def azel(eul): return eul[:,0], 90-eul[:,1]     # azimuth=rot, elevation=90-tilt (plot2d.py convention)

# ===== FIGURE 1: DIRECT pose maps (rows half1/half2; cols cryoFIFA | cryoSPARC | median error) =====
fig,ax=plt.subplots(2,3,figsize=(15.5,8))
for ri,hh in enumerate(['half1','half2']):
    af,ef=azel(cfe[hh]); ac,ec=azel(cse[hh]); r=res[hh]
    hf=ax[ri,0].hist2d(af,ef,bins=BINS,range=RANGE,cmap='turbo')
    ax[ri,1].hist2d(ac,ec,bins=BINS,range=RANGE,cmap='turbo')
    M,xe,ye,_=binned_statistic_2d(ac,ec,r,statistic='median',bins=BINS,range=RANGE)
    pe=ax[ri,2].imshow(M.T,origin='lower',extent=[-180,180,-90,90],aspect='auto',cmap='viridis_r',vmin=0,vmax=15)
    ax[ri,0].set_title(f'{hh}: cryoFIFA poses'); ax[ri,1].set_title(f'{hh}: cryoSPARC poses')
    ax[ri,2].set_title(f'{hh}: median error / view')
    for j in range(3):
        ax[ri,j].set_xlabel('Azimuth (°)'); ax[ri,j].set_ylabel('Elevation (°)')
        ax[ri,j].set_xticks([-180,-90,0,90,180]); ax[ri,j].set_yticks([-90,-45,0,45,90])
    fig.colorbar(hf[3],ax=ax[ri,1],label='Particle Count',fraction=0.046)
    fig.colorbar(pe,ax=ax[ri,2],label='median error (°)',fraction=0.046)
    fig.colorbar(hf[3],ax=ax[ri,0],label='Particle Count',fraction=0.046)
fig.suptitle('cryoFIFA vs cryoSPARC viewing-direction distributions (same aligned frame)',fontsize=14,y=1.00)
plt.tight_layout()
for ext in ('png','pdf'): plt.savefig(SP+'fig_pose_direct.'+ext,dpi=200)
plt.close()

# ===== FIGURE 2: detailed statistics =====
fig,ax=plt.subplots(1,3,figsize=(15,4.5))
xs=np.linspace(0,60,3000)
for hh in ['half1','half2','total']:
    r=res[hh]; cdf=np.searchsorted(np.sort(r),xs)/len(r)*100
    ax[0].plot(xs,cdf,lw=2,color=C[hh],label=hh)
for x in (5,10): ax[0].axvline(x,color='0.6',ls='--',lw=.8); ax[0].text(x+0.6,4,f'{x}°',color='0.4',fontsize=9)
ax[0].set_xlim(0,40); ax[0].set_ylim(0,100); ax[0].set_xlabel('angular error vs cryoSPARC (°)')
ax[0].set_ylabel('cumulative particles (%)'); ax[0].legend(frameon=False,loc='lower right'); ax[0].set_title('(a) cumulative agreement',loc='left')
rows=[[hh,f"{np.median(res[hh]):.1f}",f"{np.percentile(res[hh],25):.1f}/{np.percentile(res[hh],75):.1f}",
       f"{100*np.mean(res[hh]<5):.0f}",f"{100*np.mean(res[hh]<10):.0f}"] for hh in ['half1','half2','total']]
tbl=ax[0].table(cellText=rows,colLabels=['set','med°','Q1/Q3','<5°%','<10°%'],bbox=[0.30,0.30,0.66,0.34]); tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tb=np.linspace(0,180,19); ctr=(tb[:-1]+tb[1:])/2
for hh in ['half1','half2']:
    tilt=cse[hh][:,1]
    med,_,_=binned_statistic(tilt,res[hh],'median',bins=tb)
    q1,_,_=binned_statistic(tilt,res[hh],lambda v:np.percentile(v,25),bins=tb)
    q3,_,_=binned_statistic(tilt,res[hh],lambda v:np.percentile(v,75),bins=tb)
    ax[1].plot(ctr,med,'-o',ms=3,color=C[hh],label=hh); ax[1].fill_between(ctr,q1,q3,color=C[hh],alpha=0.15)
ax[1].set_xlabel('cryoSPARC tilt (°)'); ax[1].set_ylabel('pose error (°)'); ax[1].set_ylim(0,25); ax[1].legend(frameon=False)
ax[1].set_title('(b) error vs viewing tilt (median, IQR)',loc='left')
w=0.38
for i,hh in enumerate(['half1','half2']):
    cp=copies[hh]; r=res[hh]; med=[np.median(r[cp==c]) if (cp==c).any() else 0 for c in range(8)]
    ax[2].bar(np.arange(8)+(i-0.5)*w,med,w,color=C[hh],label=hh)
ax[2].set_xticks(range(8)); ax[2].set_xlabel('winning sym copy (0-3: rot90k; 4-7: +flip)')
ax[2].set_ylabel('median error (°)'); ax[2].legend(frameon=False); ax[2].set_title('(c) accuracy by winning copy',loc='left')
plt.tight_layout()
for ext in ('png','pdf'): plt.savefig(SP+'fig_pose_stats.'+ext,dpi=200)
plt.close()
print("saved fig_pose_direct.{png,pdf} fig_pose_stats.{png,pdf}")
for hh in ['half1','half2','total']:
    r=res[hh]; print(f"{hh}: median={np.median(r):.2f} Q1={np.percentile(r,25):.2f} Q3={np.percentile(r,75):.2f} <5={100*np.mean(r<5):.1f}% <10={100*np.mean(r<10):.1f}%")
