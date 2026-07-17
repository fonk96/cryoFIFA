#!/usr/bin/env python3
"""
plot_pose_vs_loss.py -- relate cryoFIFA per-particle best-of-8 LOSS to pose ERROR (vs cryoSPARC).
Answers: does a particle's reconstruction loss predict how well its pose is estimated? (the mechanistic
basis for loss-ranking AND for "FIFA-easy particles are better particles").

  python plot_pose_vs_loss.py <pose_star_glob> <loss_txt_glob> <cs.star> <out_prefix> [label]

Reuses pose_bestof8.py convention search + per-copy undo for res, then joins res<->loss by particle name.
Figures (Arial+SVG): hexbin(loss,err) + Spearman; %<5deg vs loss-decile (keep-lowest-loss purity curve);
loss distributions split solved(<5)/unsolved.
"""
import sys, glob, os, re
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.stats import spearmanr, pearsonr
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'; plt.rcParams['pdf.fonttype']=42
plt.rcParams['axes.linewidth']=0.8; plt.rcParams['savefig.bbox']='tight'
Mz=np.diag([1.,1,-1]); LABEL=sys.argv[5] if len(sys.argv)>5 else 'B2 half1 (norefine)'

def norm_keys(name):
    keys=[name]
    if '@' in name:
        idx,path=name.split('@',1)
        keys+=[name, os.path.basename(path)+'@'+idx.lstrip('0'), idx.lstrip('0'), os.path.basename(path)]
    keys.append(os.path.basename(name))
    m=re.search(r'(\d+)',name)
    if m: keys.append(str(int(m.group(1))))
    out=[];seen=set()
    for k in keys:
        if k not in seen: seen.add(k);out.append(k)
    return out

# per-particle loss (name -> loss)
loss={}
for fn in glob.glob(sys.argv[2]):
    for ln in open(fn):
        t=ln.split()
        if len(t)>=2:
            try: loss[t[0]]=float(t[1])
            except: pass
# FIFA poses (name -> rot,tilt,psi,copy)
cf={}
for fn in glob.glob(sys.argv[1]):
    for ln in open(fn):
        t=ln.split()
        if len(t)<14 or t[0][0] in '[_' or t[0].startswith('data') or t[0].startswith('loop'): continue
        try: cf[t[0]]=(float(t[1]),float(t[2]),float(t[3]),int(float(t[13])))
        except: pass
# cs consensus
cur={};cols={};rows=[]
for ln in open(sys.argv[3]):
    s=ln.strip()
    if not s or s.startswith('#'): continue
    if s.startswith('data_'): cur={}; continue
    if s.startswith('loop_'): continue
    if s.startswith('_'):
        tk=s.split(); cur[tk[0]]=int(tk[1][1:])-1 if len(tk)>1 and tk[1].startswith('#') else len(cur); continue
    if '_rlnAngleRot' in cur:
        tk=s.split()
        if len(tk)>=len(cur): rows.append(tk); cols=cur
cr,ct,cpc=cols['_rlnAngleRot'],cols['_rlnAngleTilt'],cols['_rlnAnglePsi']
csidx={}
for r in rows:
    for k in norm_keys(r[cols['_rlnImageName']]): csidx.setdefault(k,r)

Rwin=[];csang=[];copies=[];lval=[]
for nm,(rot,tilt,psi,cpy) in cf.items():
    if nm not in loss: continue
    row=None
    for k in norm_keys(nm):
        if k in csidx: row=csidx[k]; break
    if row is None: continue
    Rwin.append(R.from_euler('zyz',[rot,tilt,psi],degrees=True).as_matrix())
    csang.append([float(row[cr]),float(row[ct]),float(row[cpc])]); copies.append(cpy); lval.append(loss[nm])
Rwin=np.array(Rwin); csang=np.array(csang); copies=np.array(copies); lval=np.array(lval); N=len(Rwin)
print(f"matched (pose & loss & cs)={N}")

def kab(A,B):
    H=np.einsum('nij,nkj->ik',B,A); U,_,Vt=np.linalg.svd(H); d=np.sign(np.linalg.det(Vt.T@U.T)); return Vt.T@np.diag([1,1,d])@U.T
def geo(A,B):
    tr=np.einsum('nij,nij->n',A,B); return np.degrees(np.arccos(np.clip((tr-1)/2,-1,1)))
def nearest_rot(M):
    U,_,Vt=np.linalg.svd(M); d=np.sign(np.linalg.det(U@Vt)); return U@np.diag([1,1,d])@Vt
def build_cs(et,order,tr,hd):
    ang=csang if order=='rtp' else csang[:,[2,1,0]]
    B=R.from_euler(et,ang,degrees=True).as_matrix()
    if tr: B=np.transpose(B,(0,2,1))
    if hd: B=Mz[None]@B@Mz[None]
    return B
sel0=copies==0; best=None
for et in ('ZYZ','zyz'):
  for order in ('rtp','ptr'):
    for tr in (False,True):
      for hd in (False,True):
        B=build_cs(et,order,tr,hd); Rg=kab(Rwin[sel0],B[sel0])
        m=float(np.median(geo(Rwin[sel0],np.einsum('ij,njk->nik',Rg,B[sel0]))))
        if best is None or m<best[0]: best=(m,et,order,tr,hd)
_,et,order,tr,hd=best
csR=build_cs(et,order,tr,hd); Rg=kab(Rwin[sel0],csR[sel0])
Uc={}
for c in range(8):
    sel=copies==c
    if sel.sum()==0: continue
    RwT=np.transpose(Rwin[sel],(0,2,1)); Uc[c]=nearest_rot(np.einsum('nij,jk,nkl->nil',RwT,Rg,csR[sel]).mean(0))
Uarr=np.array([Uc[c] for c in copies])
res=geo(np.einsum('nij,njk->nik',Rwin,Uarr), np.einsum('ij,njk->nik',Rg,csR))
print(f"OVERALL median={np.median(res):.2f}  <5deg={100*np.mean(res<5):.1f}%")

# correlations
rho,pr=spearmanr(lval,res); pear,_=pearsonr(np.log10(lval-lval.min()+1e-6),res)
print(f"Spearman(loss,err)={rho:.3f} (p={pr:.1e})   Pearson(log-loss,err)={pear:.3f}")
np.savez(sys.argv[4]+'.npz', res=res, loss=lval, copies=copies)

# ---- figure: 3 panels ----
fig,ax=plt.subplots(1,3,figsize=(15,4.4))
# (A) hexbin loss vs error
hb=ax[0].hexbin(lval,res,gridsize=45,bins='log',cmap='viridis',mincnt=1)
ax[0].axhline(5,color='#c0392b',lw=1,ls='--')
ax[0].set_xlabel('per-particle loss (best-of-8)',fontsize=11); ax[0].set_ylabel('pose error vs cryoSPARC (deg)',fontsize=11)
ax[0].set_title(f'(A) loss vs pose error\nSpearman ρ = {rho:.2f}',fontsize=11)
ax[0].set_ylim(0,180); cb=fig.colorbar(hb,ax=ax[0]); cb.set_label('particles (log)',fontsize=9)
# (B) keep-lowest-loss purity: sort by loss, cumulative %<5 among the kept lowest-loss fraction
o=np.argsort(lval); rr=res[o]
frac=np.arange(1,N+1)/N; cum_solved=np.cumsum(rr<5)/np.arange(1,N+1)*100
ax[1].plot(frac*100,cum_solved,color='#1a7d1a',lw=2.2)
ax[1].axhline(100*np.mean(res<5),color='#888',lw=1,ls=':',label=f'all particles ({100*np.mean(res<5):.0f}%)')
ax[1].set_xlabel('keep lowest-loss X% of particles',fontsize=11); ax[1].set_ylabel('% solved (<5°) among kept',fontsize=11)
ax[1].set_title('(B) low loss enriches well-posed particles',fontsize=11); ax[1].set_xlim(0,100); ax[1].set_ylim(0,100)
ax[1].legend(frameon=False,fontsize=9,loc='lower left'); ax[1].grid(alpha=0.25)
# per-decile median error (twin)
axt=ax[1].twinx()
dec=np.clip((np.searchsorted(np.sort(lval),lval,side='right')-1)//max(1,N//10),0,9)
med=[np.median(res[dec==d]) for d in range(10)]
axt.plot(np.arange(10)*10+5,med,color='#f0932b',lw=1.6,marker='o',ms=4,alpha=0.85)
axt.set_ylabel('median error per loss-decile (deg)',fontsize=10,color='#f0932b'); axt.tick_params(axis='y',labelcolor='#f0932b'); axt.set_ylim(0,max(med)*1.15)
# (C) loss distribution solved vs unsolved
sv=res<5
lo,hi=np.percentile(lval,[0.5,99.5])
bins=np.linspace(lo,hi,50)
ax[2].hist(lval[sv],bins=bins,color='#1a7d1a',alpha=0.65,density=True,label=f'solved <5° (n={sv.sum()})')
ax[2].hist(lval[~sv],bins=bins,color='#c0392b',alpha=0.55,density=True,label=f'unsolved ≥5° (n={(~sv).sum()})')
ax[2].axvline(np.median(lval[sv]),color='#1a7d1a',lw=1.4,ls='--'); ax[2].axvline(np.median(lval[~sv]),color='#c0392b',lw=1.4,ls='--')
ax[2].set_xlabel('per-particle loss',fontsize=11); ax[2].set_ylabel('density',fontsize=11)
ax[2].set_title('(C) solved particles have lower loss',fontsize=11); ax[2].legend(frameon=False,fontsize=9)
for a in ax: a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
fig.suptitle(f'{LABEL}: per-particle loss predicts pose accuracy  (N={N})',fontsize=12.5,y=1.03)
plt.tight_layout()
OUT='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/lrab_figures/'; os.makedirs(OUT,exist_ok=True)
base=OUT+os.path.basename(sys.argv[4])
for ext in ('svg','pdf','png'): fig.savefig(base+'.'+ext,dpi=200)
print('saved',base+'.{svg,pdf,png}')
print(f'  median loss: solved={np.median(lval[sv]):.1f}  unsolved={np.median(lval[~sv]):.1f}')
print(f'  keep lowest-loss 50%%: {cum_solved[N//2-1]:.0f}%% solved  vs all {100*np.mean(res<5):.0f}%%')
