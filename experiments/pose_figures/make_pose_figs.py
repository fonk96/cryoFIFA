import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size':11,'axes.linewidth':0.8,'font.family':'DejaVu Sans',
                 'xtick.direction':'out','ytick.direction':'out','savefig.bbox':'tight'})
SP='/tmp/claude-1000/-sddn-kongfang-work-cryoFIFA-V260706-cryoFIFA/5b6a4386-0d4c-46ae-a59a-407ec3a82e9b/scratchpad/'
h1=np.load(SP+'pose_h1.npz'); h2=np.load(SP+'pose_h2.npz')
r1,r2=h1['res'],h2['res']; rT=np.concatenate([r1,r2])
v1,v2=h1['view'],h2['view']; vT=np.concatenate([v1,v2])
C={'half1':'#0072B2','half2':'#D55E00','total':'#222222'}
def stats(r): return dict(med=np.median(r),mean=r.mean(),p5=100*np.mean(r<5),p10=100*np.mean(r<10),p15=100*np.mean(r<15))
S={'half1':stats(r1),'half2':stats(r2),'total':stats(rT)}
for k in S: print(k, {a:round(b,2) for a,b in S[k].items()})

# ---------- FIGURE 1: pose accuracy ----------
fig,ax=plt.subplots(1,2,figsize=(10.5,4.2))
# (a) histogram 0-30 deg
bins=np.linspace(0,30,61)
for lab,r in [('half1',r1),('half2',r2)]:
    ax[0].hist(r,bins=bins,histtype='step',lw=1.8,color=C[lab],density=True,
               label=f"{lab} (median {S[lab]['med']:.1f}°)")
    ax[0].axvline(S[lab]['med'],color=C[lab],ls=':',lw=1.0)
ax[0].set_xlabel('angular error vs cryoSPARC (°)'); ax[0].set_ylabel('density')
ax[0].set_xlim(0,30); ax[0].legend(frameon=False,fontsize=9.5); ax[0].set_title('(a) per-particle pose error',loc='left',fontsize=11)
# (b) CDF
xs=np.linspace(0,180,4000)
for lab,r in [('half1',r1),('half2',r2),('total',rT)]:
    cdf=np.searchsorted(np.sort(r),xs)/len(r)*100
    ax[1].plot(xs,cdf,lw=2.0,color=C[lab],label=f"{lab}: {S[lab]['p10']:.0f}% <10°")
for x in (5,10):
    ax[1].axvline(x,color='0.6',ls='--',lw=0.8); ax[1].text(x+1.2,8,f'{x}°',color='0.4',fontsize=8.5)
ax[1].set_xlabel('angular error vs cryoSPARC (°)'); ax[1].set_ylabel('cumulative particles (%)')
ax[1].set_xlim(0,60); ax[1].set_ylim(0,100); ax[1].legend(frameon=False,loc='lower right',fontsize=9.5)
ax[1].set_title('(b) cumulative pose agreement',loc='left',fontsize=11)
plt.tight_layout();
for ext in ('png','pdf'): plt.savefig(SP+'fig_pose_accuracy.'+ext,dpi=220)
plt.close()

# ---------- FIGURE 2: orientation coverage (Mollweide) ----------
def lonlat(v):
    lon=np.arctan2(v[:,1],v[:,0]); lat=np.arcsin(np.clip(v[:,2],-1,1)); return lon,lat
fig=plt.figure(figsize=(13,3.9))
for i,(lab,v) in enumerate([('half1',v1),('half2',v2),('total',vT)]):
    ax=fig.add_subplot(1,3,i+1,projection='mollweide')
    lon,lat=lonlat(v)
    H,xe,ye=np.histogram2d(lon,lat,bins=[np.linspace(-np.pi,np.pi,73),np.linspace(-np.pi/2,np.pi/2,37)])
    Xc,Yc=np.meshgrid((xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2)
    pcm=ax.pcolormesh(Xc,Yc,H.T,cmap='magma',shading='auto')
    ax.set_title(f'{lab}  (n={len(v):,})',fontsize=11); ax.grid(True,lw=0.3,color='0.7')
    ax.set_xticklabels([]); ax.set_yticklabels([])
cax=fig.add_axes([0.92,0.18,0.012,0.62]); fig.colorbar(pcm,cax=cax,label='particles / bin')
fig.suptitle('Orientation coverage (projection-direction density, Mollweide)',fontsize=12,y=1.02)
plt.tight_layout(rect=[0,0,0.91,1])
for ext in ('png','pdf'): plt.savefig(SP+'fig_orientation_coverage.'+ext,dpi=220)
plt.close()
print('saved fig_pose_accuracy.{png,pdf} fig_orientation_coverage.{png,pdf}')
