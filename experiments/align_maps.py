#!/usr/bin/env python3
"""Pre-align the 12 ablation maps to a common frame (offline, so ChimeraX just tiles).
Global rotation search + Powell refine on downsampled maps, with a mirror check (handedness),
correlation vs the reference (curric_marchonly_03). Writes aligned copies to aligned_maps/."""
import os, numpy as np, mrcfile
from scipy.ndimage import affine_transform, zoom
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

BASE='results/2026_Jul_06'
OUT=os.path.join(BASE,'aligned_maps'); os.makedirs(OUT,exist_ok=True)
REF=('snr03_marchonly', f'{BASE}/curric_marchonly_03/rec_3d_70_22min.mrc')
MAPS=[('snr01_neither',f'{BASE}/curric_neither_01/rec_3d_70_42min.mrc'),
      ('snr01_lossrankonly',f'{BASE}/curric_lossrankonly_01/rec_3d_70_43min.mrc'),
      ('snr01_marchonly',f'{BASE}/curric_marchonly_01/rec_3d_20_3min.mrc'),
      ('snr01_both',f'{BASE}/curric_both_01/rec_3d_70_22min.mrc'),
      ('snr03_neither',f'{BASE}/curric_neither_03/rec_3d_70_41min.mrc'),
      ('snr03_lossrankonly',f'{BASE}/curric_lossrankonly_03/rec_3d_70_41min.mrc'),
      ('snr03_marchonly',REF[1]),
      ('snr03_both',f'{BASE}/curric_both_03/rec_3d_70_22min.mrc'),
      ('mix_neither',f'{BASE}/mix0103_neither/rec_3d_70_83min.mrc'),
      ('mix_lossrankonly',f'{BASE}/mix0103_lossrankonly/rec_3d_70_82min.mrc'),
      ('mix_marchonly',f'{BASE}/mix0103_lrOFF/rec_3d_70_43min.mrc'),
      ('mix_both',f'{BASE}/mix0103_lrON/rec_3d_70_43min.mrc')]
DS=48

def load(p):
    with mrcfile.open(p,permissive=True) as m:
        return np.asarray(m.data,dtype=np.float32).copy(), float(m.voxel_size.x)
def norm(v):
    v=v-v.mean(); s=v.std(); return v/s if s>0 else v
def ds(v): return zoom(v, DS/v.shape[0], order=1)
def rot(vol,euler,c):
    Rm=R.from_euler('ZYZ',euler,degrees=True).as_matrix()
    return affine_transform(vol, Rm.T, offset=c-Rm.T@c, order=1)
def corr(a,b): return float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

ref_full,vox=load(REF[1]); ref_ds=norm(ds(ref_full)); cds=(np.array(ref_ds.shape)-1)/2.0
grid=[(r,t,p) for r in range(0,360,45) for t in range(0,181,30) for p in range(0,360,45)]

for tag,path in MAPS:
    mv,_=load(path); best=(-2,None,False)
    for mir in (False,True):
        b_ds=norm(ds(mv[::-1] if mir else mv))
        for e in grid:
            c=corr(rot(b_ds,e,cds),ref_ds)
            if c>best[0]: best=(c,e,mir)
    c0,e0,mir=best
    b_ds=norm(ds(mv[::-1] if mir else mv))
    r=minimize(lambda e:-corr(rot(b_ds,e,cds),ref_ds), e0, method='Powell')
    ef,cf=(r.x,-r.fun) if -r.fun>c0 else (e0,c0)
    mfull=(mv[::-1].copy() if mir else mv); cfull=(np.array(mfull.shape)-1)/2.0
    aligned=rot(mfull,ef,cfull)
    with mrcfile.new(os.path.join(OUT,tag+'.mrc'),overwrite=True) as m:
        m.set_data(aligned.astype(np.float32)); m.voxel_size=vox
    print(f'{tag:20s} corr={cf:5.3f} mirror={int(mir)} euler={np.round(ef,1)}')
print('DONE ->',OUT)
