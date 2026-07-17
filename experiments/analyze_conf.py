#!/usr/bin/env python3
"""
analyze_conf.py -- which GT-FREE per-particle confidence metric best predicts pose error on B2?
Validates candidate signals (usable after ab-initio when NO GT poses exist) against the actual pose
error vs cryoSPARC. Compares best-of-8 confidence metrics to raw loss (the weak baseline, rho~0.10).

  python analyze_conf.py <pose_glob> <conf_glob> <loss_glob> <cs.star> <out_prefix>

conf_infer cols: name d1 d2 dmean dstd wmax entropy   (d1=winner dist, d2=runner-up over the 8 hypotheses)
"""
import sys, glob, os, re
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.stats import spearmanr
plt.rcParams['font.family']='Arial'; plt.rcParams['svg.fonttype']='none'; plt.rcParams['pdf.fonttype']=42
plt.rcParams['axes.linewidth']=0.8; plt.rcParams['savefig.bbox']='tight'
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

conf={}
for fn in glob.glob(sys.argv[2]):
    for ln in open(fn):
        t=ln.split()
        if len(t)>=7:
            try: conf[t[0]]=[float(x) for x in t[1:7]]
            except: pass
loss={}
for fn in glob.glob(sys.argv[3]):
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
cur={};cols={};rows=[]
for ln in open(sys.argv[4]):
    s=ln.strip()
    if not s or s.startswith('#'): continue
    if s.startswith('data_'): cur={}; continue
    if s.startswith('loop_'): continue
    if s.startswith('_'):
        tk=s.split(); cur[tk[0]]=int(tk[1][1:])-1 if len(tk)>1 and tk[1].startswith('#') else len(cur); continue
    if '_rlnAngleRot' in cur:
        tk=s.split()
        if len(tk)>=len(cur): rows.append(tk); cols=cur
cr,ct,cpc=cols['_rlnAngleRot'],cols['_rlnAngleTilt'],cols['_rlnAnglePsi']
csidx={}
for r in rows:
    for k in norm_keys(r[cols['_rlnImageName']]): csidx.setdefault(k,r)
Rwin=[];csang=[];copies=[];C=[];L=[]
for nm,(rot,tilt,psi,cpy) in cf.items():
    if nm not in conf: continue
    row=None
    for k in norm_keys(nm):
        if k in csidx: row=csidx[k]; break
    if row is None: continue
    Rwin.append(R.from_euler('zyz',[rot,tilt,psi],degrees=True).as_matrix())
    csang.append([float(row[cr]),float(row[ct]),float(row[cpc])]); copies.append(cpy); C.append(conf[nm]); L.append(loss.get(nm,np.nan))
Rwin=np.array(Rwin); csang=np.array(csang); copies=np.array(copies); C=np.array(C); L=np.array(L); N=len(Rwin)
print(f"matched (pose & conf & cs)={N}")
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
print(f"OVERALL median={np.median(res):.2f}  <5deg={100*np.mean(res<5):.1f}%")

d1,d2,dmean,dstd,wmax,ent = C[:,0],C[:,1],C[:,2],C[:,3],C[:,4],C[:,5]
eps=1e-9
# candidate metrics; 'sign'=+1 if HIGHER metric -> MORE confident (keep high), -1 if lower -> more confident
METRICS={
 'raw_loss (baseline)': (L,          -1),   # low loss -> keep
 'd1 winner-dist':      (d1,         -1),
 'margin d2-d1':        (d2-d1,      +1),   # big gap -> confident
 'rel_margin (d2-d1)/d1':((d2-d1)/(d1+eps), +1),
 'ratio d1/dmean':      (d1/(dmean+eps), -1),# winner far below mean -> confident
 'gap_norm (d2-d1)/dstd':((d2-d1)/(dstd+eps), +1),
 'winner_weight wmax':  (wmax,       +1),
 'entropy':             (ent,        -1),
}
print(f"\n{'metric':26s}{'|rho|':>8s}{'keep30%<5':>11s}{'keep50%<5':>11s}{'keep70%<5':>11s}")
print('-'*67)
results={}
sv=res<5
for name,(v,sign) in METRICS.items():
    ok=np.isfinite(v)
    rho,_=spearmanr(v[ok], res[ok])
    conf_score = sign*v   # higher = more confident
    o=np.argsort(-conf_score[ok])            # most confident first
    rr=res[ok][o]
    def keep(f):
        k=max(1,int(f*len(rr))); return 100*np.mean(rr[:k]<5)
    results[name]=(abs(rho), keep(.3),keep(.5),keep(.7), conf_score, ok)
    print(f"{name:26s}{abs(rho):8.3f}{keep(.3):10.1f}%{keep(.5):10.1f}%{keep(.7):10.1f}%")
print(f"{'(all particles)':26s}{'':8s}{100*np.mean(sv):10.1f}%{100*np.mean(sv):10.1f}%{100*np.mean(sv):10.1f}%")
np.savez(sys.argv[5]+'.npz', res=res, C=C, L=L)

# ---- figure: purity curves (keep top-confidence X%) for the best metrics vs baseline ----
ranked=sorted(results.items(), key=lambda kv:-kv[1][2])   # by keep50
top=[k for k,_ in ranked[:4]]
if 'raw_loss (baseline)' not in top: top=top[:3]+['raw_loss (baseline)']
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
xs=np.linspace(0.02,1,60)
cmap=plt.get_cmap('tab10')
for i,name in enumerate(top):
    _,_,_,_,cs_score,ok=results[name]
    o=np.argsort(-cs_score[ok]); rr=res[ok][o]
    y=[100*np.mean(rr[:max(1,int(f*len(rr)))]<5) for f in xs]
    ax[0].plot(xs*100,y,lw=2,color=cmap(i),label=f"{name} (|ρ|={results[name][0]:.2f})")
ax[0].axhline(100*np.mean(sv),color='#888',ls=':',lw=1,label=f'all ({100*np.mean(sv):.0f}%)')
ax[0].set_xlabel('keep most-confident X% of particles',fontsize=11); ax[0].set_ylabel('% solved (<5°) among kept',fontsize=11)
ax[0].set_title('(A) GT-free confidence filters vs raw loss',fontsize=11); ax[0].legend(frameon=False,fontsize=8.5,loc='upper right')
ax[0].set_xlim(0,100); ax[0].grid(alpha=0.25)
# (B) scatter of the single best metric vs pose error
bestname=ranked[0][0]; _,_,_,_,cs_score,ok=results[bestname]
ax[1].hexbin(cs_score[ok],res[ok],gridsize=45,bins='log',cmap='viridis',mincnt=1)
ax[1].axhline(5,color='#c0392b',ls='--',lw=1); ax[1].set_ylim(0,180)
ax[1].set_xlabel(f'confidence: {bestname}',fontsize=11); ax[1].set_ylabel('pose error vs cryoSPARC (deg)',fontsize=11)
ax[1].set_title(f'(B) best metric: {bestname} (|ρ|={ranked[0][1][0]:.2f})',fontsize=11)
for a in ax: a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
fig.suptitle(f'B2 half1: GT-free particle-confidence metrics vs pose error (N={N})',fontsize=12.5,y=1.02)
plt.tight_layout()
OUT='/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/lrab_figures/'; os.makedirs(OUT,exist_ok=True)
base=OUT+os.path.basename(sys.argv[5])
for ext in ('svg','pdf','png'): fig.savefig(base+'.'+ext,dpi=200)
print('\nsaved',base+'.{svg,pdf,png}')
print('BEST GT-free metric:',bestname,'-> keep50% =',f"{results[bestname][2]:.1f}% solved (vs {100*np.mean(sv):.0f}% all)")
