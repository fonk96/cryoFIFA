#!/usr/bin/env python3
"""
analyze_traj_b2.py -- B2 half1 (REAL data) per-epoch LOSS + best-of-8 CONFIDENCE trajectory vs final
pose error. Does the trajectory across ep15..90 give a stronger GT-free particle filter than the single
final-epoch snapshot (conf ρ=0.40, loss ρ=0.10)? cs star pulled locally (scratchpad).

  python analyze_traj_b2.py
"""
import glob, os, re
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.stats import spearmanr
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'; plt.rcParams['pdf.fonttype']=42
plt.rcParams['axes.linewidth']=0.8; plt.rcParams['savefig.bbox']='tight'
Mz=np.diag([1.,1,-1]); EPS=[15,30,45,60,75,90]
BASE='results/2026_Jul_06/B2_ablation_norefine_traj_ep%d_poses/'
CS='/tmp/claude-1000/-sddn-kongfang-work-cryoFIFA-V260706-cryoFIFA/5b6a4386-0d4c-46ae-a59a-407ec3a82e9b/scratchpad/B2_act_half1.star'
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
lossep={};confep={}
for ep in EPS:
    dl={};dc={}
    for fn in glob.glob((BASE%ep)+'loss_infer_rank*.txt'):
        for ln in open(fn):
            t=ln.split()
            if len(t)>=2:
                try: dl[t[0]]=float(t[1])
                except: pass
    for fn in glob.glob((BASE%ep)+'conf_infer_rank*.txt'):
        for ln in open(fn):
            t=ln.split()
            if len(t)>=7:
                try: dc[t[0]]=float(t[3])/(float(t[1])+1e-9)   # ratio dmean/d1 (higher=confident)
                except: pass
    lossep[ep]=dl; confep[ep]=dc
    print(f"ep{ep}: loss={len(dl)} conf={len(dc)}")
# ep90 poses -> res
cf={}
for fn in glob.glob((BASE%90)+'temp_pose_infer_rank*.star'):
    for ln in open(fn):
        t=ln.split()
        if len(t)<14 or t[0][0] in '[_' or t[0].startswith(('data','loop')): continue
        try: cf[t[0]]=(float(t[1]),float(t[2]),float(t[3]),int(float(t[13])))
        except: pass
cur={};cols={};rows=[]
for ln in open(CS):
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
    if not all(nm in lossep[ep] and nm in confep[ep] for ep in EPS): continue
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
L=np.array([[lossep[ep][nm] for ep in EPS] for nm in names])
Cf=np.array([[confep[ep][nm] for ep in EPS] for nm in names])
Lz=(L-L.mean(0))/L.std(0); Cz=(Cf-Cf.mean(0))/Cf.std(0)
solved=res<5
rl=[abs(spearmanr(L[:,j],res)[0]) for j in range(len(EPS))]
rc=[abs(spearmanr(Cf[:,j],res)[0]) for j in range(len(EPS))]
print("\nper-epoch |rho| vs pose error:  epoch : loss  conf")
for j,ep in enumerate(EPS): print(f"  ep{ep:2d}: {rl[j]:.3f}  {rc[j]:.3f}")
# candidate GT-free predictors (sign so higher=confident); report |rho| + keep-top-30/50 purity
def purity(score, f):
    o=np.argsort(-score); rr=res[o]; k=max(1,int(f*len(rr))); return 100*np.mean(rr[:k]<5)
cands={
 'conf ep90 (single)': Cf[:,-1],
 'conf mean(all ep)':  Cf.mean(1),
 'conf z-rank rise':   Cz[:,-1]-Cz[:,0],
 'loss ep90 (single)': -L[:,-1],
 'loss z-rank drop':   -(Lz[:,-1]-Lz[:,0]),
 'conf ep90 + z-rank rise': (Cz[:,-1]) + (Cz[:,-1]-Cz[:,0]),
 'conf+loss ep90 combo': Cz[:,-1] - Lz[:,-1],
}
print(f"\n{'GT-free predictor':28s}{'|rho|':>8s}{'keep30<5':>10s}{'keep50<5':>10s}")
print('-'*56)
best_score=None
for name,sc in sorted(cands.items(), key=lambda kv:-purity(kv[1],0.3)):
    rho=abs(spearmanr(sc,res)[0]); p30=purity(sc,.3); p50=purity(sc,.5)
    print(f"{name:28s}{rho:8.3f}{p30:9.1f}%{p50:9.1f}%")
    if best_score is None: best_score=(name,sc,rho,p30,p50)
print(f"{'(all particles)':28s}{'':8s}{100*np.mean(solved):9.1f}%{100*np.mean(solved):9.1f}%")
# ---- figure ----
fig,ax=plt.subplots(1,3,figsize=(15.5,4.4))
for a,(Z,ttl,cu) in zip(ax[:2],[(Lz,'(A) loss z-trajectory','loss'),(Cz,'(B) confidence z-trajectory','conf')]):
    ms=Z[solved].mean(0); us=Z[~solved].mean(0)
    a.plot(EPS,ms,'-o',color='#1a7d1a',label=f'solved <5° (n={solved.sum()})')
    a.plot(EPS,us,'-o',color='#c0392b',label=f'unsolved ≥5° (n={(~solved).sum()})')
    a.axhline(0,color='#999',lw=0.7,ls=':'); a.set_xlabel('epoch',fontsize=11)
    a.set_ylabel(f'per-epoch normalized {cu} (z)',fontsize=11); a.set_title(ttl,fontsize=11); a.legend(frameon=False,fontsize=8.5)
ax[2].plot(EPS,rl,'-o',color='#c0392b',label='loss')
ax[2].plot(EPS,rc,'-o',color='#1a7d1a',label='confidence (d̄/d₁)')
ax[2].set_xlabel('epoch',fontsize=11); ax[2].set_ylabel('|ρ| vs final pose error',fontsize=11)
ax[2].set_title('(C) predictive power per epoch',fontsize=11); ax[2].legend(frameon=False,fontsize=9); ax[2].grid(alpha=0.25); ax[2].set_ylim(0,0.5)
for a in ax: a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
fig.suptitle(f'B2 half1 (real): per-epoch loss & confidence trajectory vs pose accuracy (N={N})',fontsize=12.5,y=1.02)
plt.tight_layout()
OUT='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/lrab_figures/'
for ext in ('svg','pdf','png'): fig.savefig(OUT+'b2_traj.'+ext,dpi=200)
print('\nsaved',OUT+'b2_traj.{svg,pdf,png}')
print('BEST GT-free predictor:',best_score[0],f'-> |rho|={best_score[2]:.2f}, keep30={best_score[3]:.0f}% (vs single conf ep90 & all {100*np.mean(solved):.0f}%)')
