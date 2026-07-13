import sys, glob, os, re
import numpy as np
from scipy.spatial.transform import Rotation as R
Mx=np.diag([-1.,1,1]); My=np.diag([1.,-1,1]); Mz=np.diag([1.,1,-1])

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

# cryoFIFA best-of-8: name->(rot,tilt,psi,copy)
cf={}
for fn in glob.glob(sys.argv[1]):
    for ln in open(fn):
        t=ln.split()
        if len(t)<14 or t[0][0] in '[_' or t[0].startswith('data') or t[0].startswith('loop'): continue
        try: cf[t[0]]=(float(t[1]),float(t[2]),float(t[3]),int(float(t[13])))
        except: pass
# J68
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
cr,ct,cp=cols['_rlnAngleRot'],cols['_rlnAngleTilt'],cols['_rlnAnglePsi']
Rwin=[];csang=[];copies=[]
for nm,(rot,tilt,psi,cpy) in cf.items():
    row=None
    for k in norm_keys(nm):
        if k in csidx: row=csidx[k]; break
    if row is None: continue
    Rwin.append(R.from_euler('zyz',[rot,tilt,psi],degrees=True).as_matrix())
    csang.append([float(row[cr]),float(row[ct]),float(row[cp])]); copies.append(cpy)
Rwin=np.array(Rwin); csang=np.array(csang); copies=np.array(copies)
csR=R.from_euler('ZYZ',csang,degrees=True).as_matrix()
print(f"matched={len(Rwin)}  copy dist={np.bincount(copies,minlength=8)}")

def kab(A,B):
    H=np.einsum('nij,nkj->ik',B,A); U,_,Vt=np.linalg.svd(H); d=np.sign(np.linalg.det(Vt.T@U.T)); return Vt.T@np.diag([1,1,d])@U.T
def geo(A,B):
    tr=np.einsum('nij,nij->n',A,B); return np.degrees(np.arccos(np.clip((tr-1)/2,-1,1)))
def best_conv(A,B):   # search cs euler/transpose/hand, return (median, params, residual)
    best=None
    for et in ('ZYZ','zyz'):
      for order in ('rtp','ptr'):
        for tr in (False,True):
          for hd in (False,True):
            ang=csang if order=='rtp' else csang[:,[2,1,0]]
            Bm=R.from_euler(et,ang,degrees=True).as_matrix()
            if tr: Bm=np.transpose(Bm,(0,2,1))
            if hd: Bm=Mz[None]@Bm@Mz[None]
            Bm=Bm[B]; Am=A
            Rg=kab(Am,Bm); res=geo(Am,np.einsum('ij,njk->nik',Rg,Bm)); m=float(np.median(res))
            if best is None or m<best[0]: best=(m,(et,order,tr,hd),res)
    return best

# TEST 1: copy-0 subset (no undo needed) -- decisive
sel0=copies==0
print(f"\n=== TEST 1: copy-0 subset ({sel0.sum()} particles, NO undo) ===")
m,par,res=best_conv(Rwin[sel0], np.where(sel0)[0])
print(f"  best cs-conv={par}  median={m:.2f}deg  <5deg={100*np.mean(res<5):.1f}%  <10deg={100*np.mean(res<10):.1f}%")

# TEST 2: data-driven per-copy undo. One fixed 3x3 per copy fit to align that copy to cryoSPARC;
# a single transform aligns a whole copy ONLY if the poses are truly accurate (not circular).
def nearest_rot(M):
    U,_,Vt=np.linalg.svd(M); d=np.sign(np.linalg.det(U@Vt)); return U@np.diag([1,1,d])@Vt
sel0=copies==0
Rg=kab(Rwin[sel0], csR[sel0])                       # global frame from the clean copy-0 subset
print("\n=== TEST 2: data-driven per-copy undo (cs conv ZYZ[rot,tilt,psi]) ===")
allres=[]
for c in range(8):
    sel=copies==c
    if sel.sum()==0: continue
    RwT=np.transpose(Rwin[sel],(0,2,1))
    Uc=nearest_rot(np.einsum('nij,jk,nkl->nil', RwT, Rg, csR[sel]).mean(0))   # Uc = <Rwin^T Rg csR>
    res=geo(np.einsum('nij,jk->nik',Rwin[sel],Uc), np.einsum('ij,njk->nik',Rg,csR[sel]))
    allres.append(res)
    print(f"  copy {c} (n={sel.sum():5d}): median={np.median(res):5.2f}deg  <5deg={100*np.mean(res<5):3.0f}%  det(Uc)={np.linalg.det(Uc):+.0f}")
allres=np.concatenate(allres)
print(f"  OVERALL (50k): median={np.median(allres):.2f}deg  <5deg={100*np.mean(allres<5):.1f}%  <10deg={100*np.mean(allres<10):.1f}%  <15deg={100*np.mean(allres<15):.1f}%")
