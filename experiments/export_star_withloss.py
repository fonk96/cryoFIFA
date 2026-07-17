#!/usr/bin/env python3
"""
export_star_withloss.py -- annotate a cryoSPARC/RELION star with cryoFIFA per-particle LOSS (and pose
error vs cryoSPARC) so particles can be GLOBALLY post-filtered by loss (GT-free: loss is defined for
every particle). Writes the FULL particle set with two appended columns _cryofifaLoss + _cryofifaPoseErr,
and also a lowest-loss-kept subset for convenience.

  python export_star_withloss.py <pose_star_glob> <loss_txt_glob> <cs.star> <out.star> [keep_frac=1.0]

keep_frac<1.0 also writes <out>.keepNN.star = the lowest-loss keep_frac fraction (global loss filter).
Prints the loss percentile table so you can pick a threshold. Pose error is FIFA-vs-cryoSPARC (needs the
consensus star); the loss column does NOT need it -> use loss for a fresh/GT-free dataset.
"""
import sys, glob, os, re
import numpy as np
from scipy.spatial.transform import Rotation as R
Mz=np.diag([1.,1,-1]); KEEP=float(sys.argv[5]) if len(sys.argv)>5 else 1.0

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

loss={}
for fn in glob.glob(sys.argv[2]):
    for ln in open(fn):
        t=ln.split()
        if len(t)>=2:
            try: loss[t[0]]=float(t[1])
            except: pass
cf={}
for fn in glob.glob(sys.argv[1]):
    for ln in open(fn):
        t=ln.split()
        if len(t)<14 or t[0][0] in '[_' or t[0].startswith('data') or t[0].startswith('loop'): continue
        try: cf[t[0]]=(float(t[1]),float(t[2]),float(t[3]),int(float(t[13])))
        except: pass
raw=open(sys.argv[3]).read().splitlines()
cur={};cols={};rows=[]
for ln in raw:
    s=ln.strip()
    if not s or s.startswith('#'): continue
    if s.startswith('data_'): cur={}; continue
    if s.startswith('loop_'): continue
    if s.startswith('_'):
        tk=s.split(); cur[tk[0]]=int(tk[1][1:])-1 if len(tk)>1 and tk[1].startswith('#') else len(cur); continue
    if '_rlnAngleRot' in cur:
        tk=s.split()
        if len(tk)>=len(cur): rows.append(tk); cols=cur
img=cols['_rlnImageName']; cr,ct,cpc=cols['_rlnAngleRot'],cols['_rlnAngleTilt'],cols['_rlnAnglePsi']
csidx={}
for r in rows:
    for k in norm_keys(r[img]): csidx.setdefault(k,r)
Rwin=[];csang=[];copies=[];ll=[];names=[]
for nm,(rot,tilt,psi,cpy) in cf.items():
    row=None
    for k in norm_keys(nm):
        if k in csidx: row=csidx[k]; break
    if row is None: continue
    Rwin.append(R.from_euler('zyz',[rot,tilt,psi],degrees=True).as_matrix())
    csang.append([float(row[cr]),float(row[ct]),float(row[cpc])]); copies.append(cpy)
    ll.append(loss.get(nm,-1.0)); names.append(row[img])
Rwin=np.array(Rwin); csang=np.array(csang); copies=np.array(copies); ll=np.array(ll); N=len(Rwin)
print(f"matched={N}  with_loss={int((ll>=0).sum())}")

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
# per-cs-name lookup of (loss, err)
info={names[i]:(ll[i],res[i]) for i in range(N)}

# ---- write full annotated star: append _cryofifaLoss + _cryofifaPoseErr ----
def write_star(path, keep_names=None):
    out=[]; in_particles=False; hdr_done=False; ncol=len(cols); wrote=0
    for ln in raw:
        s=ln.strip()
        if s.startswith('data_particles'): in_particles=True
        is_hdr = in_particles and s.startswith('_')
        is_row = in_particles and bool(s) and not s.startswith(('#','_','data','loop'))
        if in_particles and not hdr_done and is_row:
            out.append('_cryofifaLoss #%d'%(ncol+1)); out.append('_cryofifaPoseErr #%d'%(ncol+2)); hdr_done=True
        if is_row:
            tk=s.split()
            if len(tk)>=ncol:
                nm=tk[img]
                if keep_names is not None and nm not in keep_names: continue
                lv,ev=info.get(nm,(-1.0,-1.0))
                out.append(ln.rstrip()+'  %.4f %.3f'%(lv,ev)); wrote+=1; continue
        out.append(ln)
    open(path,'w').write('\n'.join(out)+'\n'); return wrote

w=write_star(sys.argv[4])
print(f"wrote FULL annotated star: {w} particles -> {sys.argv[4]}")
# loss percentile guide
have=ll[ll>=0]
print("loss percentiles:", {int(p):round(float(np.percentile(have,p)),1) for p in (5,10,25,50,75,90,95)})
for q in (0.3,0.5,0.7):
    thr=np.quantile(have,q)
    print(f"  keep lowest-loss {int(q*100)}% (loss<={thr:.1f}): {int((have<=thr).sum())} particles, %<5deg among kept={100*np.mean(res[ll>=0][have<=thr]<5):.1f}")
if KEEP<1.0:
    o=np.argsort(ll); thr=np.quantile(have,KEEP)
    keep_names=set(names[i] for i in range(N) if 0<=ll[i]<=thr)
    kp=sys.argv[4].replace('.star','')+'.keep%02d.star'%int(KEEP*100)
    kw=write_star(kp, keep_names)
    print(f"wrote LOW-LOSS subset (keep {int(KEEP*100)}%): {kw} particles -> {kp}")
