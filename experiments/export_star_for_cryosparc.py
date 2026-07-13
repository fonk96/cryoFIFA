#!/usr/bin/env python3
"""
export_star_for_cryosparc.py -- convert cryoFIFA best-of-8 poses to a RELION-3.1 star that cryoSPARC
can import and 'reconstruct only'. Expresses the winning (undone) pose in cryoSPARC's own frame/
convention (Rcf_cs = Rg^T Rcf, angles = as_euler('ZYZ'), same as the J68 star cryoSPARC produced), so
both halves share one frame (combinable, gold-standard). Shift is the winning translation with the
per-copy 90deg/flip undone + RELION sign; verify/refine shifts in cryoSPARC if the map looks soft.

  python export_star_for_cryosparc.py <cs_J68.star> <out.star> <fifa_h1_glob> [<fifa_h2_glob>]
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
    keys.append(os.path.basename(name)); m=re.search(r'(\d+)',name)
    if m: keys.append(str(int(m.group(1))))
    out=[];seen=set()
    for k in keys:
        if k not in seen: seen.add(k); out.append(k)
    return out
def kab(A,B):
    H=np.einsum('nij,nkj->ik',B,A); U,_,Vt=np.linalg.svd(H); d=np.sign(np.linalg.det(Vt.T@U.T)); return Vt.T@np.diag([1,1,d])@U.T
def geo(A,B):
    tr=np.einsum('nij,nij->n',A,B); return np.degrees(np.arccos(np.clip((tr-1)/2,-1,1)))
def nearest(M):
    U,_,Vt=np.linalg.svd(M); d=np.sign(np.linalg.det(U@Vt)); return U@np.diag([1,1,d])@Vt

cs_path, out_path = sys.argv[1], sys.argv[2]
# parse J68 particles block
cur={};cols={};rows=[]
for ln in open(cs_path):
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
CR,CT,CP=cols['_rlnAngleRot'],cols['_rlnAngleTilt'],cols['_rlnAnglePsi']

def R2(deg):
    t=np.radians(deg); c,s=np.cos(t),np.sin(t); return np.array([[c,-s],[s,c]])

def process(fifa_glob, subset):
    Rwin=[];csang=[];copies=[];meta=[]
    for fn in glob.glob(fifa_glob):
        for ln in open(fn):
            t=ln.split()
            if len(t)<14 or t[0][0] in '[_' or t[0].startswith('data') or t[0].startswith('loop'): continue
            try: rot,tilt,psi=float(t[1]),float(t[2]),float(t[3]); cpy=int(float(t[13]))
            except: continue
            row=None
            for k in norm_keys(t[0]):
                if k in csidx: row=csidx[k]; break
            if row is None: continue
            Rwin.append(R.from_euler('zyz',[rot,tilt,psi],degrees=True).as_matrix())
            csang.append([float(row[CR]),float(row[CT]),float(row[CP])]); copies.append(cpy)
            # meta: name, dfu,dfv,dfang,volt,cs,ampc, ox,oy, apix
            meta.append((t[0],float(t[4]),float(t[5]),float(t[6]),float(t[7]),float(t[8]),float(t[9]),float(t[10]),float(t[11]),float(t[12])))
    Rwin=np.array(Rwin); csang=np.array(csang); copies=np.array(copies)
    csR=R.from_euler('ZYZ',csang,degrees=True).as_matrix()
    sel0=copies==0; Rg=kab(Rwin[sel0],csR[sel0])
    Uc={c:nearest(np.einsum('nij,jk,nkl->nil',np.transpose(Rwin[copies==c],(0,2,1)),Rg,csR[copies==c]).mean(0)) for c in range(8) if (copies==c).any()}
    Rcf=np.einsum('nij,njk->nik',Rwin,np.array([Uc[c] for c in copies]))
    Rcfcs=np.einsum('ij,njk->nik',Rg.T,Rcf)                # cryoFIFA pose in cryoSPARC frame
    res=geo(Rcfcs,csR); print(f"  subset {subset}: n={len(Rcf)} median vs cryoSPARC={np.median(res):.2f}deg")
    ang=R.from_matrix(Rcfcs).as_euler('ZYZ',degrees=True)  # rot,tilt,psi
    out=[]
    for i,(nm,dfu,dfv,dfa,volt,cs,ampc,ox,oy,apix) in enumerate(meta):
        # cryoFIFA's shift predictions do NOT correlate with cryoSPARC's (r~0 even for copy-0) -> they are
        # unreliable, so we DON'T use them: OriginAngst=0 (reconstruct at the extraction centers, cleaner
        # than injecting random shifts). Let cryoSPARC re-estimate shifts in a local refinement.
        oxa,oya=0.0,0.0
        out.append((nm,ang[i,0],ang[i,1],ang[i,2],oxa,oya,dfu,dfv,dfa,subset))
    return out, meta[0]

allrows=[]; optics=None
optics=process.__doc__  # placeholder
rows_all=[]
for gi,glb in enumerate(sys.argv[3:]):
    r,m0=process(glb, gi+1); rows_all+=r
    if optics is None or gi==0: opt=m0
volt,cs,ampc,apix=opt[4],opt[5],opt[6],opt[9]
with open(out_path,'w') as f:
    f.write("\ndata_optics\n\nloop_\n_rlnOpticsGroup #1\n_rlnOpticsGroupName #2\n_rlnVoltage #3\n_rlnSphericalAberration #4\n_rlnAmplitudeContrast #5\n_rlnImagePixelSize #6\n_rlnImageSize #7\n_rlnImageDimensionality #8\n")
    f.write(f"1 opticsGroup1 {volt} {cs} {ampc} {apix} 228 2\n")
    f.write("\ndata_particles\n\nloop_\n_rlnImageName #1\n_rlnAngleRot #2\n_rlnAngleTilt #3\n_rlnAnglePsi #4\n_rlnOriginXAngst #5\n_rlnOriginYAngst #6\n_rlnDefocusU #7\n_rlnDefocusV #8\n_rlnDefocusAngle #9\n_rlnRandomSubset #10\n_rlnOpticsGroup #11\n")
    for nm,rot,tilt,psi,oxa,oya,dfu,dfv,dfa,sub in rows_all:
        f.write(f"{nm} {rot:.4f} {tilt:.4f} {psi:.4f} {oxa:.3f} {oya:.3f} {dfu:.1f} {dfv:.1f} {dfa:.3f} {sub} 1\n")
print(f"wrote {out_path}: {len(rows_all)} particles")
