#!/usr/bin/env python3
"""
vol_align_fsc.py -- align two independent ab-initio half-maps, compute gold-standard FSC, merge.

Rigidly aligns half2 onto half1 (rotational cross-correlation search over SO(3), both hands, then
local refinement), computes FSC(half1, half2_aligned) masked + unmasked, writes the aligned half2,
a soft mask, and the merged (averaged) map. Self-contained: numpy + scipy only.

Usage:
  python vol_align_fsc.py --h1 <half1.mrc> --h2 <half2.mrc> --apix <A/px> --out <prefix>
"""
import argparse, numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, binary_dilation
from scipy.spatial.transform import Rotation as R

def read_mrc(fn):
    with open(fn,'rb') as f:
        h=f.read(1024)
        nx,ny,nz=np.frombuffer(h,np.int32,3)
        mode=int(np.frombuffer(h,np.int32,1,12)[0]); ns=int(np.frombuffer(h,np.int32,1,92)[0])
        f.seek(1024+ns); dt={0:np.int8,1:np.int16,2:np.float32,6:np.uint16}[mode]
        d=np.frombuffer(f.read(),dt).astype(np.float32)
    return d.reshape(nz,ny,nx)

def write_mrc(fn,v,apix):
    v=np.asarray(v,dtype=np.float32); n=v.shape[0]
    h=bytearray(1024)
    def pi(off,*vals):
        for i,val in enumerate(vals): h[off+i*4:off+i*4+4]=np.int32(val).tobytes()
    def pf(off,*vals):
        for i,val in enumerate(vals): h[off+i*4:off+i*4+4]=np.float32(val).tobytes()
    pi(0,n,n,n,2)                       # NX NY NZ MODE(float32)
    pi(28,n,n,n)                        # MX MY MZ
    pf(40,n*apix,n*apix,n*apix)         # CELLA (Angstrom) -> apix = CELLA/MX
    pf(52,90,90,90)                     # CELLB angles
    pi(64,1,2,3)                        # MAPC MAPR MAPS
    h[208:212]=b'MAP '                  # format stamp
    with open(fn,'wb') as f: f.write(bytes(h)); f.write(v.tobytes())

def downsample(v,n):
    F=np.fft.fftshift(np.fft.fftn(v)); c=v.shape[0]//2; h=n//2
    Fc=F[c-h:c+h,c-h:c+h,c-h:c+h]
    return np.fft.ifftn(np.fft.ifftshift(Fc)).real.astype(np.float32)

def rotate_vol(v,Rm,order=1):
    c=(np.array(v.shape)-1)/2.0
    return affine_transform(v, Rm.T, offset=c-Rm.T@c, order=order, mode='constant', cval=0.0)

def ncc(a,b):
    a=a-a.mean(); b=b-b.mean(); d=np.sqrt((a*a).sum()*(b*b).sum())+1e-9
    return float((a*b).sum()/d)

def best_rotation(h1,h2,seed=0):
    """coarse SO(3) search on 48^3 (both hands), refine on 96^3 for precision."""
    a48=downsample(h1,48); a48=(a48-a48.mean())/(a48.std()+1e-9)
    a96=downsample(h1,96); a96=(a96-a96.mean())/(a96.std()+1e-9)
    results=[]
    for hand,vh in ((+1,h2),(-1,h2[:,:,::-1])):     # try mirror (ab-initio hand ambiguity)
        b48=downsample(vh,48); b96=downsample(vh,96)
        best=(-1,None)
        for Rm in R.random(6000,random_state=seed).as_matrix():
            c=ncc(a48,rotate_vol(b48,Rm))
            if c>best[0]: best=(c,Rm)
        cur=(ncc(a96,rotate_vol(b96,best[1])),best[1])   # refine on 96^3
        for it in range(5):
            sig=0.15/(it+1)
            for Rm in R.from_rotvec(np.random.RandomState(seed*13+it).normal(0,sig,(400,3))).as_matrix()@cur[1]:
                c=ncc(a96,rotate_vol(b96,Rm))
                if c>cur[0]: cur=(c,Rm)
        results.append((cur[0],hand,cur[1]))
        print(f"  hand {'+' if hand>0 else '-'}: NCC(96^3) = {cur[0]:.4f}")
    results.sort(reverse=True); return results[0]   # (ncc, hand, R)

def fsc(a,b):
    Fa=np.fft.fftshift(np.fft.fftn(a)); Fb=np.fft.fftshift(np.fft.fftn(b))
    n=a.shape[0]; c=n//2
    z,y,x=[q-c for q in np.indices(a.shape)]; r=np.sqrt(x*x+y*y+z*z).astype(int)
    num=np.bincount(r.ravel(), (Fa*np.conj(Fb)).real.ravel())
    da=np.bincount(r.ravel(), (np.abs(Fa)**2).ravel()); db=np.bincount(r.ravel(), (np.abs(Fb)**2).ravel())
    f=num/(np.sqrt(da*db)+1e-9)
    return f[:c]

def res_at(fsc_curve,n,apix,thr):
    for i in range(1,len(fsc_curve)):
        if fsc_curve[i-1]>=thr and fsc_curve[i]<thr:
            t=(fsc_curve[i-1]-thr)/(fsc_curve[i-1]-fsc_curve[i]+1e-9); fi=i-1+t
            return n*apix/fi if fi>0 else None
    return None

def soft_mask(v,apix,lp=15,thr_sigma=2.0,dilate=3,soft=3):
    n=v.shape[0]; vs=gaussian_filter(v, sigma=max(1.0,(lp/apix)/3.0))
    m=(vs> vs.mean()+thr_sigma*vs.std()).astype(np.float32)
    m=binary_dilation(m, iterations=dilate).astype(np.float32)
    m=gaussian_filter(m, sigma=soft); m=np.clip(m/max(m.max(),1e-6),0,1)
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--h1',required=True); ap.add_argument('--h2',required=True)
    ap.add_argument('--apix',type=float,required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args()
    h1=read_mrc(a.h1); h2=read_mrc(a.h2); n=h1.shape[0]
    print(f"box={n} apix={a.apix}")
    score,hand,Rm=best_rotation(h1,h2)
    print(f"BEST: hand={'flip' if hand<0 else 'same'}  NCC={score:.4f}")
    h2s = h2[:,:,::-1] if hand<0 else h2
    h2a = rotate_vol(h2s, Rm, order=3)     # cubic for the full-res apply -> preserve high freq
    write_mrc(a.out+'_half2_aligned.mrc', h2a, a.apix)
    mask=soft_mask(h1,a.apix); write_mrc(a.out+'_mask.mrc',mask,a.apix)
    fu=fsc(h1,h2a); fm=fsc(h1*mask,h2a*mask)
    print("\nFSC (unmasked / masked):")
    for thr in (0.5,0.143):
        ru=res_at(fu,n,a.apix,thr); rm=res_at(fm,n,a.apix,thr)
        su='%.2fA'%ru if ru else '>Nyq'; sm='%.2fA'%rm if rm else '>Nyq'
        print(f"  FSC={thr}:  unmasked {su}   masked {sm}")
    print("  masked FSC curve:")
    for i in range(2,len(fm),max(1,len(fm)//12)):
        print(f"    {n*a.apix/i:6.2f}A : {fm[i]:+.3f}")
    merged=0.5*(h1+h2a); write_mrc(a.out+'_merged.mrc',merged,a.apix)
    def kurt(v): z=(v-v.mean())/(v.std()+1e-9); return float((z**4).mean()-3)
    print(f"\nmerged kurtosis={kurt(merged):.1f}  (half1={kurt(h1):.1f} half2={kurt(h2a):.1f})")
    print(f"wrote: {a.out}_half2_aligned.mrc  {a.out}_mask.mrc  {a.out}_merged.mrc")

if __name__=='__main__': main()
