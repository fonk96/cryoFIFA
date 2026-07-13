import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size':15,'axes.linewidth':1.0,'font.family':'DejaVu Sans','savefig.bbox':'tight'})
SP='/tmp/claude-1000/-sddn-kongfang-work-cryoFIFA-V260706-cryoFIFA/5b6a4386-0d4c-46ae-a59a-407ec3a82e9b/scratchpad/'
h1=np.load(SP+'pose_h1.npz'); h2=np.load(SP+'pose_h2.npz')
RANGE=[[-180,180],[-90,90]]; BINS=64
def azel(eul): return eul[:,0], 90.0-eul[:,1]     # azimuth=rot, elevation=90-tilt (plot2d.py)

def viewfig(az,el,title,fn):
    plt.figure(figsize=(9,5))
    plt.hist2d(az,el,bins=BINS,range=RANGE,cmap='turbo')
    plt.title(title,fontsize=16)
    plt.xlabel('Azimuth (°)',fontsize=15); plt.ylabel('Elevation (°)',fontsize=15)
    plt.xticks([-180,-90,0,90,180]); plt.yticks([-90,-45,0,45,90])
    cb=plt.colorbar(label='Particle Count'); cb.ax.tick_params(labelsize=12)
    plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(SP+fn+'.'+ext,dpi=220)
    plt.close()

# 1) half1 cryoFIFA viewing distribution   2) half2   3) cryoSPARC (both in cryoSPARC frame)
a,e=azel(h1['cf_eul_cs']); viewfig(a,e,'half1 — cryoFIFA viewing distribution','fig1_half1_viewdist')
a,e=azel(h2['cf_eul_cs']); viewfig(a,e,'half2 — cryoFIFA viewing distribution','fig2_half2_viewdist')
csn=np.concatenate([h1['cs_eul_native'],h2['cs_eul_native']])
a,e=azel(csn); viewfig(a,e,'cryoSPARC viewing distribution','fig3_cryosparc_viewdist')

# 4) pose error statistics -- single plot (CDF, half1/half2/total)
r1,r2=h1['res'],h2['res']; rT=np.concatenate([r1,r2])
C={'half1':'#0072B2','half2':'#D55E00','total':'#222222'}
plt.figure(figsize=(7.5,5.5))
xs=np.linspace(0,60,3000)
for lab,r in [('half1',r1),('half2',r2),('total',rT)]:
    cdf=np.searchsorted(np.sort(r),xs)/len(r)*100
    plt.plot(xs,cdf,lw=2.4,color=C[lab],
             label=f"{lab}: median {np.median(r):.1f}°, {100*np.mean(r<10):.0f}% <10°")
for x in (5,10): plt.axvline(x,color='0.6',ls='--',lw=1.0); plt.text(x+0.7,4,f'{x}°',color='0.4',fontsize=12)
plt.xlim(0,40); plt.ylim(0,100)
plt.xlabel('pose error vs cryoSPARC (°)',fontsize=15); plt.ylabel('cumulative particles (%)',fontsize=15)
plt.title('cryoFIFA pose accuracy',fontsize=16); plt.legend(frameon=False,loc='lower right',fontsize=12.5)
plt.grid(alpha=0.25); plt.tight_layout()
for ext in ('png','pdf'): plt.savefig(SP+'fig4_pose_error.'+ext,dpi=220)
plt.close()
print("saved fig1_half1_viewdist fig2_half2_viewdist fig3_cryosparc_viewdist fig4_pose_error  (.png/.pdf)")
