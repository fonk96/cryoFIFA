#!/usr/bin/env python3
"""
analyze_mix.py -- split-by-SNR pose accuracy for mixed-SNR runs.

  python analyze_mix.py <fifa_star_glob> <gt.star> <high_stack> [out.npz]

Identifies each particle's SNR source by the mrcs stack in its rlnImageName
(<high_stack> = the high-SNR stack, e.g. J530 or J532; everything else = low-SNR).
The cryoSPARC-convention detection, global Kabsch frame, and per-copy undo are ALL
fit on the HIGH-SNR subset (the solvable particles), then applied to everyone, so the
low-SNR junk cannot bias the reference frame. Reports OVERALL / HIGH / LOW accuracy.
Mirrors pose_bestof8.py conventions exactly.
"""
import sys, glob, os, re
import numpy as np
from scipy.spatial.transform import Rotation as R
Mz=np.diag([1.,1,-1])
HIGH=sys.argv[3]

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

cf={}
for fn in glob.glob(sys.argv[1]):
    for ln in open(fn):
        t=ln.split()
        if len(t)<14 or t[0][0] in '[_' or t[0].startswith('data') or t[0].startswith('loop'): continue
        try: cf[t[0]]=(float(t[1]),float(t[2]),float(t[3]),int(float(t[13])))
        except: pass
cur={};cols={};rows=[]
for ln in open(sys.argv[2]):
    s=ln.strip()
    if not s or s.startswith('#'): continue
    if s.startswith('data_'): cur={}; continue
    if s.startswith('loop_'): continue
    if s.startswith('_'):
        tk=s.split(); cur[tk[0]]=int(tk[1][1:])-1 if len(tk)>1 and tk[1].startswith('#') else len(cur); continue
    if '_rlnAngleRot' in cur:
        tk=s.split()
        if len(tk)>=len(cur): rows.append(tk); cols=cur
csidx={}
for r in rows:
    for k in norm_keys(r[cols['_rlnImageName']]): csidx.setdefault(k,r)
cr,ct,cpc=cols['_rlnAngleRot'],cols['_rlnAngleTilt'],cols['_rlnAnglePsi']
Rwin=[];csang=[];copies=[];ishigh=[]
for nm,(rot,tilt,psi,cpy) in cf.items():
    row=None
    for k in norm_keys(nm):
        if k in csidx: row=csidx[k]; break
    if row is None: continue
    Rwin.append(R.from_euler('zyz',[rot,tilt,psi],degrees=True).as_matrix())
    csang.append([float(row[cr]),float(row[ct]),float(row[cpc])]); copies.append(cpy)
    ishigh.append(HIGH in nm)
Rwin=np.array(Rwin); csang=np.array(csang); copies=np.array(copies); hi=np.array(ishigh); N=len(Rwin)
print(f"matched={N}  high({HIGH})={hi.sum()}  low={(~hi).sum()}  copy dist={np.bincount(copies,minlength=8)}")

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

# --- fit convention + frame on HIGH-SNR copy-0 subset (solvable reference) ---
sel0=(copies==0)&hi
if sel0.sum()<10: sel0=(copies==0)   # fallback if too few high-SNR copy-0
best=None
for et in ('ZYZ','zyz'):
  for order in ('rtp','ptr'):
    for tr in (False,True):
      for hd in (False,True):
        B=build_cs(et,order,tr,hd)
        Rg=kab(Rwin[sel0],B[sel0]); m=float(np.median(geo(Rwin[sel0],np.einsum('ij,njk->nik',Rg,B[sel0]))))
        if best is None or m<best[0]: best=(m,et,order,tr,hd)
_,et,order,tr,hd=best
csR=build_cs(et,order,tr,hd)
Rg=kab(Rwin[sel0],csR[sel0])
print(f"cs convention: {et}[{'rot,tilt,psi' if order=='rtp' else 'psi,tilt,rot'}] transpose={tr} hand={hd}  (fit on n={sel0.sum()} high-SNR copy0)")

# per-copy undo fit on HIGH-SNR winners of each copy (low-SNR junk excluded)
Uc={}
for c in range(8):
    sel=(copies==c)&hi
    if sel.sum()<5: continue
    RwT=np.transpose(Rwin[sel],(0,2,1))
    Uc[c]=nearest_rot(np.einsum('nij,jk,nkl->nil',RwT,Rg,csR[sel]).mean(0))
Ident=np.eye(3)
Uarr=np.array([Uc.get(c,Ident) for c in copies])
Rcf=np.einsum('nij,njk->nik',Rwin,Uarr)
Rcs=np.einsum('ij,njk->nik',Rg,csR)
res=geo(Rcf,Rcs)

def report(tag,m):
    if m.sum()==0: print(f"{tag:6s}: (none)"); return
    r=res[m]
    print(f"{tag:6s} (n={m.sum():5d}): median={np.median(r):6.2f}  mean={r.mean():6.2f}  <5deg={100*np.mean(r<5):5.1f}%  <10deg={100*np.mean(r<10):5.1f}%")
report("ALL",np.ones(N,bool))
report("HIGH",hi)
report("LOW",~hi)
if len(sys.argv)>4:
    np.savez(sys.argv[4], res=res, copies=copies, ishigh=hi)
    print("saved",sys.argv[4])
