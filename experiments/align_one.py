#!/usr/bin/env python3
"""Align a single map to the montage reference (curric_marchonly_03) with mirror check + search + refine.
  python align_one.py <in_map.mrc> <out_name>   -> writes results/2026_Jul_06/aligned_maps/<out_name>.mrc"""
import sys, os, numpy as np, mrcfile
from scipy.ndimage import affine_transform, zoom
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
B='results/2026_Jul_06'; OUT=os.path.join(B,'aligned_maps'); os.makedirs(OUT,exist_ok=True)
REF=f'{B}/curric_marchonly_03/rec_3d_70_22min.mrc'
INMAP=sys.argv[1]; OUTNAME=sys.argv[2]; DS=48
def load(p):
    with mrcfile.open(p,permissive=True) as m: return np.asarray(m.data,np.float32).copy(), float(m.voxel_size.x)
def nrm(v): v=v-v.mean(); s=v.std(); return v/s if s>0 else v
def ds(v): return zoom(v,DS/v.shape[0],order=1)
def rot(v,e,c):
    Rm=R.from_euler('ZYZ',e,degrees=True).as_matrix(); return affine_transform(v,Rm.T,offset=c-Rm.T@c,order=1)
def cc(a,b): return float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
ref,vox=load(REF); ref_ds=nrm(ds(ref)); c=(np.array(ref_ds.shape)-1)/2.0
grid=[(a,t,p) for a in range(0,360,45) for t in range(0,181,30) for p in range(0,360,45)]
mv,_=load(INMAP); best=(-2,None,False)
for mir in (False,True):
    bd=nrm(ds(mv[::-1] if mir else mv))
    for e in grid:
        v=cc(rot(bd,e,c),ref_ds)
        if v>best[0]: best=(v,e,mir)
c0,e0,mir=best; bd=nrm(ds(mv[::-1] if mir else mv))
r=minimize(lambda e:-cc(rot(bd,e,c),ref_ds),e0,method='Powell'); ef=r.x if -r.fun>c0 else e0
mf=(mv[::-1].copy() if mir else mv); cf=(np.array(mf.shape)-1)/2.0
al=rot(mf,ef,cf)
with mrcfile.new(os.path.join(OUT,OUTNAME+'.mrc'),overwrite=True) as m:
    m.set_data(al.astype(np.float32)); m.voxel_size=vox
print(f'{OUTNAME}: corr={-r.fun if -r.fun>c0 else c0:.3f} mirror={int(mir)} euler={np.round(ef,1)} -> {OUT}/{OUTNAME}.mrc')
