import numpy as np, os, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size':13,'font.family':'DejaVu Sans','savefig.bbox':'tight'})
SP='/tmp/claude-1000/-sddn-kongfang-work-cryoFIFA-V260706-cryoFIFA/5b6a4386-0d4c-46ae-a59a-407ec3a82e9b/scratchpad/'
SNR={'005':0.05,'01':0.1,'02':0.2,'03':0.3,'04':0.4,'05':0.5}
RANGE=[[-180,180],[-90,90]]; BINS=48
def azel(e): return e[:,0], 90-e[:,1]
curve=[]
for tag,s in sorted(SNR.items(),key=lambda x:x[1]):
    f=SP+f'snr_{tag}.npz'
    if not os.path.exists(f): continue
    d=np.load(f); r=d['res']; curve.append((s,np.median(r),np.percentile(r,25),np.percentile(r,75),100*np.mean(r<5),100*np.mean(r<10)))
    # per-SNR: cryoFIFA viewdist | ground-truth viewdist | error CDF
    fig,ax=plt.subplots(1,3,figsize=(15,4.3))
    af,ef=azel(d['cf_eul_cs']); h=ax[0].hist2d(af,ef,bins=BINS,range=RANGE,cmap='turbo'); fig.colorbar(h[3],ax=ax[0],label='count')
    ax[0].set_title(f'cryoFIFA poses (SNR {s})')
    ag,eg=azel(d['cs_eul_native']); h2=ax[1].hist2d(ag,eg,bins=BINS,range=RANGE,cmap='turbo'); fig.colorbar(h2[3],ax=ax[1],label='count')
    ax[1].set_title('ground-truth poses')
    for a in ax[:2]: a.set_xlabel('Azimuth (°)'); a.set_ylabel('Elevation (°)'); a.set_xticks([-180,0,180]); a.set_yticks([-90,0,90])
    xs=np.linspace(0,180,2000); cdf=np.searchsorted(np.sort(r),xs)/len(r)*100
    ax[2].plot(xs,cdf,lw=2.2,color='#0072B2'); ax[2].axvline(5,ls='--',c='0.6',lw=.8); ax[2].axvline(10,ls='--',c='0.6',lw=.8)
    ax[2].set_xlim(0,60); ax[2].set_ylim(0,100); ax[2].grid(alpha=.3)
    ax[2].set_xlabel('pose error vs ground truth (°)'); ax[2].set_ylabel('cumulative particles (%)')
    ax[2].set_title(f'median {np.median(r):.1f}°  |  {100*np.mean(r<10):.0f}% <10°')
    plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(SP+f'snr_{tag}_pose.{ext}',dpi=180)
    plt.close(); print(f"SNR {s}: median {np.median(r):.2f}deg  <5={100*np.mean(r<5):.0f}%  <10={100*np.mean(r<10):.0f}%  -> snr_{tag}_pose.png")
# accuracy-vs-SNR summary curve
if curve:
    c=np.array(curve); s=c[:,0]
    fig,ax=plt.subplots(1,2,figsize=(11,4.6))
    ax[0].fill_between(s,c[:,2],c[:,3],alpha=0.18,color='#0072B2'); ax[0].plot(s,c[:,1],'-o',color='#0072B2',lw=2.2,ms=7)
    ax[0].set_xlabel('SNR'); ax[0].set_ylabel('pose error vs ground truth (°)'); ax[0].set_title('(a) median pose error (IQR)',loc='left'); ax[0].grid(alpha=.3); ax[0].set_ylim(bottom=0)
    ax[1].plot(s,c[:,4],'-o',color='#0072B2',lw=2.2,ms=7,label='within 5°'); ax[1].plot(s,c[:,5],'-s',color='#D55E00',lw=2.2,ms=7,label='within 10°')
    ax[1].set_xlabel('SNR'); ax[1].set_ylabel('particles (%)'); ax[1].set_ylim(0,100); ax[1].grid(alpha=.3); ax[1].legend(frameon=False); ax[1].set_title('(b) fraction accurately posed',loc='left')
    plt.tight_layout()
    for ext in ('png','pdf'): plt.savefig(SP+f'fig_snr_accuracy.{ext}',dpi=220)
    print("saved fig_snr_accuracy.{png,pdf}")
