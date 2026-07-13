#!/usr/bin/env python3
"""
pose_bestof8.py -- compare cryoFIFA best-of-8 poses to cryoSPARC (J68), save per-particle data.

  python pose_bestof8.py <fifa_star_glob> <cs.star> <out.npz>

cryoFIFA star cols: name rot tilt psi dfu dfv dfang volt cs ampc ox oy apix SymCopy(0-7).
Finds cs convention from the copy-0 subset (no undo), then a data-driven per-copy undo (one fixed
rotation per winning copy), reports median angular error, saves res/viewdir/copies to out.npz.
"""
import sys, glob, os, re
import numpy as np
from scipy.spatial.transform import Rotation as R
Mz=np.diag([1.,1,-1])

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
Rwin=[];csang=[];copies=[]
for nm,(rot,tilt,psi,cpy) in cf.items():
    row=None
    for k in norm_keys(nm):
        if k in csidx: row=csidx[k]; break
    if row is None: continue
    Rwin.append(R.from_euler('zyz',[rot,tilt,psi],degrees=True).as_matrix())
    csang.append([float(row[cr]),float(row[ct]),float(row[cpc])]); copies.append(cpy)
Rwin=np.array(Rwin); csang=np.array(csang); copies=np.array(copies); N=len(Rwin)
print(f"matched={N}  copy dist={np.bincount(copies,minlength=8)}")

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

sel0=copies==0
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
print(f"cs convention: {et}[{'rot,tilt,psi' if order=='rtp' else 'psi,tilt,rot'}] transpose={tr} hand={hd}")

Uc={}
for c in range(8):
    sel=copies==c
    if sel.sum()==0: continue
    RwT=np.transpose(Rwin[sel],(0,2,1))
    Uc[c]=nearest_rot(np.einsum('nij,jk,nkl->nil',RwT,Rg,csR[sel]).mean(0))
Uarr=np.array([Uc[c] for c in copies])
Rcf=np.einsum('nij,njk->nik',Rwin,Uarr)
Rcs=np.einsum('ij,njk->nik',Rg,csR)
res=geo(Rcf,Rcs)
# conventional viewing (projection) direction from cryoSPARC (rot,tilt) -> both halves in the SAME frame
_rot=np.radians(csang[:,0]); _tilt=np.radians(csang[:,1])
view=np.stack([np.sin(_tilt)*np.cos(_rot), np.sin(_tilt)*np.sin(_rot), np.cos(_tilt)],1)
for c in range(8):
    sel=copies==c
    print(f"  copy {c} (n={sel.sum():5d}): median={np.median(res[sel]):5.2f}deg")
print(f"OVERALL: median={np.median(res):.2f}  mean={res.mean():.2f}  <5deg={100*np.mean(res<5):.1f}%  <10deg={100*np.mean(res<10):.1f}%")
# Save eulers in cryoSPARC's NATIVE frame so half1/half2/cryoSPARC distributions are all comparable.
def eul(M): return R.from_matrix(M).as_euler('ZYZ',degrees=True)   # [N,3]=rot,tilt,psi
Rcfcs=np.einsum('ij,njk->nik',Rg.T,Rcf)                            # cryoFIFA pose in cryoSPARC frame
np.savez(sys.argv[3], res=res, copies=copies,
         cf_eul_cs=eul(Rcfcs), cs_eul_native=csang,                # both in cryoSPARC native frame
         cf_eul=eul(Rcf), cs_eul=eul(Rcs))
print("saved",sys.argv[3])
