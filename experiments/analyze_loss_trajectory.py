#!/usr/bin/env python3
"""
analyze_loss_trajectory.py -- does per-particle loss ACROSS epochs separate well-solved (<5deg) from
ill-solved better than a single final-epoch snapshot? Proof-of-concept on mix0103_lrON (per-epoch loss
already exported, ep10..70), SNR0.3 subpop (J530) scored vs its own GT. Loss magnitude changes each epoch
(marching), so normalize per-epoch (z-score) before comparing trajectories.

  python analyze_loss_trajectory.py
"""
import glob, os, re
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.stats import spearmanr
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'; plt.rcParams['pdf.fonttype']=42
plt.rcParams['axes.linewidth']=0.8; plt.rcParams['savefig.bbox']='tight'
Mz=np.diag([1.,1,-1])
BASE='results/2026_Jul_06/'; EPS=[10,20,30,40,50,60,70]
GT='/sddn/kongfang_work/cryoFIFA/data1/ribo_snr03.star'; SUBPOP='@J530'
def norm_keys(name):
    keys=[name]
    if '@' in name:
        idx,path=name.split('@',1); keys+=[name, os.path.basename(path)+'@'+idx.lstrip('0'), idx.lstrip('0'), os.path.basename(path)]
    keys.append(os.path.basename(name)); m=re.search(r'(\d+)',name)
    if m: keys.append(str(int(m.group(1))))
    out=[];seen=set()
    for k in keys:
        if k not in seen: seen.add(k);out.append(k)
    return out
# per-epoch loss
lossep={}
for ep in EPS:
    d={}
    for fn in glob.glob(f'{BASE}mix0103_lrON_loss_ep{ep}/loss_infer_rank*.txt'):
        for ln in open(fn):
            t=ln.split()
            if len(t)>=2 and SUBPOP in t[0]:
                try: d[t[0]]=float(t[1])
                except: pass
    lossep[ep]=d
    print(f"ep{ep}: {len(d)} {SUBPOP} particles")
# ep70 poses (winner) for SNR0.3 -> res vs GT
cf={}
for fn in glob.glob(f'{BASE}mix0103_lrON_loss_ep70/temp_pose_infer_rank*.star'):
    for ln in open(fn):
        t=ln.split()
        if len(t)<14 or t[0][0] in '[_' or t[0].startswith(('data','loop')) or SUBPOP not in t[0]: continue
        try: cf[t[0]]=(float(t[1]),float(t[2]),float(t[3]),int(float(t[13])))
        except: pass
cur={};cols={};rows=[]
for ln in open(GT):
    s=ln.strip()
    if not s or s.startswith('#'): continue
    if s.startswith('data_'): cur={};continue
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
names=[];Rwin=[];csang=[];copies=[]
for nm,(rot,tilt,psi,cpy) in cf.items():
    if not all(nm in lossep[ep] for ep in EPS): continue
    row=None
    for k in norm_keys(nm):
        if k in csidx: row=csidx[k]; break
    if row is None: continue
    names.append(nm); Rwin.append(R.from_euler('zyz',[rot,tilt,psi],degrees=True).as_matrix())
    csang.append([float(row[cr]),float(row[ct]),float(row[cpc])]); copies.append(cpy)
Rwin=np.array(Rwin);csang=np.array(csang);copies=np.array(copies);N=len(names)
def kab(A,B):
    H=np.einsum('nij,nkj->ik',B,A);U,_,Vt=np.linalg.svd(H);d=np.sign(np.linalg.det(Vt.T@U.T));return Vt.T@np.diag([1,1,d])@U.T
def geo(A,B):
    tr=np.einsum('nij,nij->n',A,B);return np.degrees(np.arccos(np.clip((tr-1)/2,-1,1)))
def nearest_rot(M):
    U,_,Vt=np.linalg.svd(M);d=np.sign(np.linalg.det(U@Vt));return U@np.diag([1,1,d])@Vt
def build_cs(et,order,tr,hd):
    ang=csang if order=='rtp' else csang[:,[2,1,0]];B=R.from_euler(et,ang,degrees=True).as_matrix()
    if tr:B=np.transpose(B,(0,2,1))
    if hd:B=Mz[None]@B@Mz[None]
    return B
sel0=copies==0;best=None
for et in('ZYZ','zyz'):
 for order in('rtp','ptr'):
  for tr in(False,True):
   for hd in(False,True):
    B=build_cs(et,order,tr,hd);Rg=kab(Rwin[sel0],B[sel0]);m=float(np.median(geo(Rwin[sel0],np.einsum('ij,njk->nik',Rg,B[sel0]))))
    if best is None or m<best[0]:best=(m,et,order,tr,hd)
_,et,order,tr,hd=best;csR=build_cs(et,order,tr,hd);Rg=kab(Rwin[sel0],csR[sel0])
Uc={}
for c in range(8):
    sel=copies==c
    if sel.sum()==0:continue
    RwT=np.transpose(Rwin[sel],(0,2,1));Uc[c]=nearest_rot(np.einsum('nij,jk,nkl->nil',RwT,Rg,csR[sel]).mean(0))
Uarr=np.array([Uc[c] for c in copies])
res=geo(np.einsum('nij,njk->nik',Rwin,Uarr),np.einsum('ij,njk->nik',Rg,csR))
print(f"N={N}  <5deg={100*np.mean(res<5):.1f}%")
# loss matrix [N, nep] + per-epoch z-score
L=np.array([[lossep[ep][nm] for ep in EPS] for nm in names])
Lz=(L-L.mean(0))/L.std(0)
solved=res<5
# per-epoch predictive power
rho_ep=[abs(spearmanr(L[:,j],res)[0]) for j in range(len(EPS))]
print("\n|rho|(loss_ep, final pose error) per epoch:")
for ep,r in zip(EPS,rho_ep): print(f"  ep{ep}: {r:.3f}")
# trajectory features
feats={'final loss (ep70)':L[:,-1],'early loss (ep10)':L[:,0],'mean loss (all ep)':L.mean(1),
       'loss drop ep10->70':-(L[:,0]-L[:,-1]),'z-rank drop':-(Lz[:,0]-Lz[:,-1]),
       'mean z-rank':Lz.mean(1),'z-rank std over epochs':Lz.std(1)}
print("\ntrajectory-feature |rho| vs final pose error:")
for k,v in sorted(feats.items(),key=lambda kv:-abs(spearmanr(kv[1],res)[0])):
    print(f"  {k:24s}{abs(spearmanr(v,res)[0]):.3f}")
# ---- figure ----
fig,ax=plt.subplots(1,2,figsize=(11,4.4))
ms=Lz[solved].mean(0); us=Lz[~solved].mean(0)
es=Lz[solved].std(0)/np.sqrt(solved.sum()); eu=Lz[~solved].std(0)/np.sqrt((~solved).sum())
ax[0].plot(EPS,ms,'-o',color='#1a7d1a',label=f'solved <5° (n={solved.sum()})')
ax[0].fill_between(EPS,ms-es,ms+es,color='#1a7d1a',alpha=0.2)
ax[0].plot(EPS,us,'-o',color='#c0392b',label=f'unsolved ≥5° (n={(~solved).sum()})')
ax[0].fill_between(EPS,us-eu,us+eu,color='#c0392b',alpha=0.2)
ax[0].axhline(0,color='#999',lw=0.7,ls=':')
ax[0].set_xlabel('epoch',fontsize=11); ax[0].set_ylabel('per-epoch normalized loss (z)',fontsize=11)
ax[0].set_title('(A) loss trajectory: solved vs unsolved',fontsize=11); ax[0].legend(frameon=False,fontsize=9)
ax[1].plot(EPS,rho_ep,'-o',color='#2c3e50')
ax[1].set_xlabel('epoch',fontsize=11); ax[1].set_ylabel('|ρ|(loss, final pose error)',fontsize=11)
ax[1].set_title('(B) when does loss become predictive?',fontsize=11); ax[1].grid(alpha=0.25); ax[1].set_ylim(0,max(rho_ep)*1.2)
for a in ax: a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
fig.suptitle(f'mix0103 lrON, SNR0.3 subpop: per-epoch loss trajectory vs final pose accuracy (N={N})',fontsize=12,y=1.02)
plt.tight_layout()
OUT='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/mix_figures/'; os.makedirs(OUT,exist_ok=True)
for ext in('svg','pdf','png'): fig.savefig(OUT+'loss_trajectory_mix.'+ext,dpi=200)
print('\nsaved',OUT+'loss_trajectory_mix.{svg,pdf,png}')
