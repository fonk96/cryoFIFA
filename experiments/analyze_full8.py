#!/usr/bin/env python3
"""
analyze_full8.py -- does the FULL 8 per-copy loss vector (with symmetry/handedness structure) predict
pose error better than the summary stats (single d1/dmean AUC~0.69, summary-combo ~0.70)? Copies 0-3 =
rot 0/90/180/270, 4-7 = their mirror-flips. Tests handedness features + a model on all 8.
  python analyze_full8.py
"""
import glob, os, re
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import mannwhitneyu
Mz=np.diag([1.,1,-1])
D='results/2026_Jul_06/B2_ablation_norefine_full8_poses/'
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
full8={}
for fn in glob.glob(D+'full8_infer_rank*.txt'):
    for ln in open(fn):
        t=ln.split()
        if len(t)>=9:
            try: full8[t[0]]=np.array([float(x) for x in t[1:9]])
            except: pass
print("full8 particles:",len(full8))
cf={}
for fn in glob.glob(D+'temp_pose_infer_rank*.star'):
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
names=[];Rwin=[];csang=[];copies=[];D8=[]
for nm,(rot,tilt,psi,cpy) in cf.items():
    if nm not in full8: continue
    row=None
    for k in norm_keys(nm):
        if k in csidx: row=csidx[k]; break
    if row is None: continue
    names.append(nm); Rwin.append(R.from_euler('zyz',[rot,tilt,psi],degrees=True).as_matrix())
    csang.append([float(row[cr]),float(row[ct]),float(row[cpc])]); copies.append(cpy); D8.append(full8[nm])
Rwin=np.array(Rwin);csang=np.array(csang);copies=np.array(copies);D8=np.array(D8);N=len(names)
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
y=(res<5).astype(float)
print(f"N={N}  <5deg={100*y.mean():.1f}%")
def auc(sc):
    U,_=mannwhitneyu(sc[y==1],sc[y==0],alternative='two-sided');a=U/((y==1).sum()*(y==0).sum());return max(a,1-a)
# per-particle normalized 8 (ratio to mean); winner + handedness features
mean8=D8.mean(1,keepdims=True); r8=D8/(mean8+1e-9)          # ratios, winner<1
wcopy=D8.argmin(1); wmin=D8.min(1)
mirror=(wcopy+4)%8                                          # each copy's mirror-flip
mirror_loss=D8[np.arange(N),mirror]
srt=np.sort(D8,1)
feats={
 'd̄/d1 (winner conf)':      mean8[:,0]/(wmin+1e-9),
 'winner mirror gap':        (mirror_loss-wmin)/(mean8[:,0]+1e-9),      # winner vs ITS mirror -> handedness
 'rotmin vs flipmin gap':    np.abs(D8[:,:4].min(1)-D8[:,4:].min(1))/(mean8[:,0]+1e-9),
 '2nd-min gap (d2-d1)/mean': (srt[:,1]-srt[:,0])/(mean8[:,0]+1e-9),
 'spread std/mean':          D8.std(1)/(mean8[:,0]+1e-9),
}
print(f"\n{'feature':28s}{'AUC':>7s}")
for k,v in sorted(feats.items(),key=lambda kv:-auc(kv[1])): print(f"  {k:28s}{auc(v):.3f}")
# LDA on all 8 self-normalized ratios (sorted, to be pose-invariant) + on 8 copy-order
def lda_auc(Xf):
    Xz=(Xf-Xf.mean(0))/(Xf.std(0)+1e-9); tr=np.arange(N)%2==0
    mu1=Xz[tr&(y==1)].mean(0);mu0=Xz[tr&(y==0)].mean(0);Sw=np.cov(Xz[tr].T)+1e-3*np.eye(Xf.shape[1])
    w=np.linalg.solve(Sw,mu1-mu0);s=Xz@w;te=~tr
    U,_=mannwhitneyu(s[te][y[te]==1],s[te][y[te]==0],alternative='two-sided');a=U/((y[te]==1).sum()*(y[te]==0).sum());return max(a,1-a)
print(f"\nLDA(8 sorted ratios):        {lda_auc(np.sort(r8,1)):.3f}")
print(f"LDA(8 copy-order ratios):    {lda_auc(r8):.3f}")
print(f"LDA(all handedness feats):   {lda_auc(np.array([feats[k] for k in feats]).T):.3f}")
print(f"\nBASELINE single d̄/d1 AUC={auc(mean8[:,0]/(wmin+1e-9)):.3f}; summary-combo was 0.70")
