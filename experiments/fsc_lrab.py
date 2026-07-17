#!/usr/bin/env python3
"""FSC-to-GT for the 3 local-refine ablation arms (align each to the GT ribosome, then FSC)."""
import numpy as np, mrcfile
from scipy.ndimage import affine_transform, zoom
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
B='results/2026_Jul_06'; APIX=3.76875; GT='snr_figures/final_maps/ground_truth.mrc'
MAPS=[('standard    (scale -)  ', f'{B}/lrab_ribo_standard/rec_3d_70_22min.mrc'),
      ('coarseonly  (scale 0)  ', f'{B}/lrab_ribo_coarseonly/rec_3d_70_23min.mrc'),
      ('scale0.05             ', f'{B}/lrab_ribo_scale0.05/rec_3d_70_23min.mrc'),
      ('scale0.1              ', f'{B}/lrab_ribo_scale0.1/rec_3d_70_23min.mrc'),
      ('coarselocal (scale 0.3)', f'{B}/lrab_ribo_coarselocal/rec_3d_70_23min.mrc')]
DS=48
def load(p):
    with mrcfile.open(p,permissive=True) as m: return np.asarray(m.data,np.float32).copy()
def nrm(v): v=v-v.mean(); s=v.std(); return v/s if s>0 else v
def ds(v): return zoom(v,DS/v.shape[0],order=1)
def rot(v,e,c):
    Rm=R.from_euler('ZYZ',e,degrees=True).as_matrix(); return affine_transform(v,Rm.T,offset=c-Rm.T@c,order=1)
def cc(a,b): return float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
def fsc(a,b):
    A=np.fft.fftshift(np.fft.fftn(a)); Bf=np.fft.fftshift(np.fft.fftn(b))
    N=a.shape[0]; ct=N//2; zz,yy,xx=np.indices(a.shape)-ct; r=np.sqrt(xx*xx+yy*yy+zz*zz).astype(int); out=[]
    for k in range(1,ct):
        m=(r==k); num=np.real(np.sum(A[m]*np.conj(Bf[m]))); den=np.sqrt(np.sum(np.abs(A[m])**2)*np.sum(np.abs(Bf[m])**2))
        out.append(num/den if den>0 else 0.0)
    return np.array(out)
def res_at(f,thr,N):
    for k,val in enumerate(f):
        if val<thr: return N*APIX/max(k,1)
    return 2*APIX
gt=load(GT); gt_ds=nrm(ds(gt)); c=(np.array(gt_ds.shape)-1)/2.0; N=gt.shape[0]
grid=[(a,t,p) for a in range(0,360,45) for t in range(0,181,30) for p in range(0,360,45)]
print(f"{'arm':13s} {'CCC(GT)':>8s} {'FSC0.5':>8s} {'FSC0.143':>9s}  mirror")
for tag,path in MAPS:
    mv=load(path); best=(-2,None,False)
    for mir in (False,True):
        bd=nrm(ds(mv[::-1] if mir else mv))
        for e in grid:
            v=cc(rot(bd,e,c),gt_ds)
            if v>best[0]: best=(v,e,mir)
    c0,e0,mir=best; bd=nrm(ds(mv[::-1] if mir else mv))
    r=minimize(lambda e:-cc(rot(bd,e,c),gt_ds),e0,method='Powell'); ef=r.x if -r.fun>c0 else e0
    mf=(mv[::-1].copy() if mir else mv); cf=(np.array(mf.shape)-1)/2.0
    al=rot(mf,ef,cf); f=fsc(nrm(al),nrm(gt))
    print(f"{tag:13s} {cc(nrm(al),nrm(gt)):8.3f} {res_at(f,0.5,N):7.1f}A {res_at(f,0.143,N):8.1f}A   {int(mir)}")
