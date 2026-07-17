#!/usr/bin/env python3
"""
export_star_for_cryosparc.py -- cryoFIFA best-of-8 poses -> RELION-3.1 star for cryoSPARC import + reconstruct.
Poses in cryoSPARC frame (Rcf_cs=Rg^T Rcf). Per-copy shift conversion is FIT to cryoSPARC (best possible),
since cryoFIFA's raw shift head is only ~0.3-correlated; translation_3d is already in pixels (phase_shift
uses dx/img_size). Shift mode via argv[4]: 'fit' (data-driven best) or 'zero' (extraction centers).

  python export_star_for_cryosparc.py <cs_J68.star> <out.star> <fit|zero> <fifa_h1_glob> [<fifa_h2_glob>]
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

cs_path, out_path, shift_mode = sys.argv[1], sys.argv[2], sys.argv[3]
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
OX,OY=cols['_rlnOriginXAngst'],cols['_rlnOriginYAngst']

def process(fifa_glob, subset):
    Rwin=[];csang=[];copies=[];meta=[];cssh=[]
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
            cssh.append([float(row[OX]),float(row[OY])])
            meta.append((t[0],float(t[4]),float(t[5]),float(t[6]),float(t[7]),float(t[8]),float(t[9]),float(t[10]),float(t[11]),float(t[12])))
    Rwin=np.array(Rwin); csang=np.array(csang); copies=np.array(copies); cssh=np.array(cssh)
    cfsh=np.array([[m[7],m[8]] for m in meta])                # cryoFIFA winning shift (px)
    csR=R.from_euler('ZYZ',csang,degrees=True).as_matrix()
    sel0=copies==0; Rg=kab(Rwin[sel0],csR[sel0])
    Uc={c:nearest(np.einsum('nij,jk,nkl->nil',np.transpose(Rwin[copies==c],(0,2,1)),Rg,csR[copies==c]).mean(0)) for c in range(8) if (copies==c).any()}
    Rcf=np.einsum('nij,njk->nik',Rwin,np.array([Uc[c] for c in copies]))
    Rcfcs=np.einsum('ij,njk->nik',Rg.T,Rcf)
    print(f"  subset {subset}: n={len(Rcf)} rot median vs cryoSPARC={np.median(geo(Rcfcs,csR)):.2f}deg")
    ang=R.from_matrix(Rcfcs).as_euler('ZYZ',degrees=True)
    # shift: per-copy least-squares fit cs_Angst ~ [cf_px,1] @ W   (best-possible conversion incl undo)
    sh=np.zeros((len(meta),2))
    if shift_mode=='fit':
        for c in range(8):
            m=copies==c
            if m.sum()<10: continue
            X=np.hstack([cfsh[m],np.ones((m.sum(),1))]); W,_,_,_=np.linalg.lstsq(X,cssh[m],rcond=None)
            sh[m]=X@W
        cc=np.corrcoef(sh[:,0],cssh[:,0])[0,1]
        print(f"    shift fit corr (pred vs cryoSPARC): {cc:.3f}  (raw cryoFIFA shift head is weak)")
    elif shift_mode=='ortho':
        # PRINCIPLED per-copy transform: undo the winning sym copy's image rotation/flip on the in-plane
        # shift with a constrained ORTHOGONAL (rotation+reflection) map -- copies 0-3 come out as R2(90k),
        # copies 4-7 as a reflection of those (validated: raw single-transform corr 0.06 -> per-copy ~0.24).
        apix_l=float(meta[0][9])                 # cryoFIFA shift is in px at box 228; OriginXAngst is in Angstrom
        cfsh=cfsh*apix_l
        raw=np.corrcoef(cfsh[:,0],cssh[:,0])[0,1]
        for c in range(8):
            m=copies==c
            if m.sum()<10: continue
            A=cfsh[m]; B=cssh[m]; Am=A.mean(0); Bm=B.mean(0)
            U,_,Vt=np.linalg.svd((A-Am).T@(B-Bm)); Q=U@Vt   # O(2): rotation (det+1) or reflection (det-1)
            sh[m]=(A-Am)@Q + Bm
            qang=np.degrees(np.arctan2(Q[1,0],Q[0,0])); flip='+flip' if np.linalg.det(Q)<0 else '     '
            print(f"      copy {c}: n={m.sum():5d}  rot={qang:+7.1f} {flip}")
        cc=np.corrcoef(sh[:,0],cssh[:,0])[0,1]
        print(f"    ortho per-copy shift corr: {cc:.3f}  (raw copy-frame corr was {raw:.3f})")
    out=[(meta[i][0],ang[i,0],ang[i,1],ang[i,2],sh[i,0],sh[i,1],meta[i][1],meta[i][2],meta[i][3],subset) for i in range(len(meta))]
    return out, meta[0]

rows_all=[]; opt=None
for gi,glb in enumerate(sys.argv[4:]):
    r,m0=process(glb, gi+1); rows_all+=r
    if opt is None: opt=m0
volt,cs,ampc,apix=opt[4],opt[5],opt[6],opt[9]
with open(out_path,'w') as f:
    f.write("\ndata_optics\n\nloop_\n_rlnOpticsGroup #1\n_rlnOpticsGroupName #2\n_rlnVoltage #3\n_rlnSphericalAberration #4\n_rlnAmplitudeContrast #5\n_rlnImagePixelSize #6\n_rlnImageSize #7\n_rlnImageDimensionality #8\n")
    f.write(f"1 opticsGroup1 {volt} {cs} {ampc} {apix} 228 2\n")
    f.write("\ndata_particles\n\nloop_\n_rlnImageName #1\n_rlnAngleRot #2\n_rlnAngleTilt #3\n_rlnAnglePsi #4\n_rlnOriginXAngst #5\n_rlnOriginYAngst #6\n_rlnDefocusU #7\n_rlnDefocusV #8\n_rlnDefocusAngle #9\n_rlnRandomSubset #10\n_rlnOpticsGroup #11\n")
    for nm,rot,tilt,psi,oxa,oya,dfu,dfv,dfa,sub in rows_all:
        f.write(f"{nm} {rot:.4f} {tilt:.4f} {psi:.4f} {oxa:.3f} {oya:.3f} {dfu:.1f} {dfv:.1f} {dfa:.3f} {sub} 1\n")
print(f"wrote {out_path}: {len(rows_all)} particles (shift_mode={shift_mode})")
