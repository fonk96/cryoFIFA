#!/usr/bin/env python3
"""
export_star_withconf.py -- annotate a star with cryoFIFA best-of-8 CONFIDENCE (winner_weight = softmax
weight of the winning pose hypothesis; higher = more confident). GT-FREE particle filter validated on B2
(rho=0.40 vs pose error, keep-top-30% -> 68% <5deg). Also emits a keep-top-X% (highest-confidence) subset.

  python export_star_withconf.py <conf_txt_glob> <cs.star> <out.star> [keep_frac=0.5]

conf cols: name d1 d2 dmean dstd wmax entropy. Adds _cryofifaConf (=wmax) + _cryofifaConfRatio (=dmean/d1,
higher=better). keep_frac<1 -> <out>.top<NN>.star = the highest-confidence keep_frac fraction.
"""
import sys, glob, os, re
import numpy as np
KEEP=float(sys.argv[4]) if len(sys.argv)>4 else 0.5

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

# conf keyed by norm_keys -> (wmax, ratio=dmean/d1)
confk={}
n_conf=0
for fn in glob.glob(sys.argv[1]):
    for ln in open(fn):
        t=ln.split()
        if len(t)>=7:
            try: d1,d2,dmean,dstd,wmax,ent=[float(x) for x in t[1:7]]
            except: continue
            ratio=dmean/(d1+1e-9); n_conf+=1
            for k in norm_keys(t[0]): confk.setdefault(k,(wmax,ratio))
print(f"conf particles read={n_conf}")

raw=open(sys.argv[2]).read().splitlines()
cur={};cols={}
for ln in raw:
    s=ln.strip()
    if not s or s.startswith('#'): continue
    if s.startswith('data_'): cur={}; continue
    if s.startswith('loop_'): continue
    if s.startswith('_'):
        tk=s.split(); cur[tk[0]]=int(tk[1][1:])-1 if len(tk)>1 and tk[1].startswith('#') else len(cur)
    elif '_rlnAngleRot' in cur: cols=cur; break
img=cols['_rlnImageName']; ncol=len(cols)
def look(nm):
    for k in norm_keys(nm):
        if k in confk: return confk[k]
    return (-1.0,-1.0)

# collect all particle confidences to compute the keep threshold
wvals=[]
for ln in raw:
    s=ln.strip()
    if s and not s.startswith(('#','_','data','loop')):
        tk=s.split()
        if len(tk)>=ncol: wvals.append(look(tk[img])[0])
wvals=np.array(wvals); have=wvals[wvals>=0]
thr=np.quantile(have, 1.0-KEEP) if KEEP<1.0 else -np.inf   # keep the TOP keep_frac by wmax -> wmax>=thr
print(f"annotated {len(wvals)} rows ({int((wvals>=0).sum())} with conf); wmax pct:",
      {int(p):round(float(np.percentile(have,p)),4) for p in (5,25,50,75,95)})

def write(path, top_only):
    out=[]; in_p=False; hdr_done=False; wrote=0
    for ln in raw:
        s=ln.strip()
        if s.startswith('data_particles'): in_p=True
        is_row = in_p and s and not s.startswith(('#','_','data','loop'))
        if in_p and not hdr_done and is_row:
            out.append('_cryofifaConf #%d'%(ncol+1)); out.append('_cryofifaConfRatio #%d'%(ncol+2)); hdr_done=True
        if is_row:
            tk=s.split()
            if len(tk)>=ncol:
                w,rt=look(tk[img])
                if top_only and not (w>=thr): continue
                out.append(ln.rstrip()+'  %.6f %.5f'%(w,rt)); wrote+=1; continue
        out.append(ln)
    open(path,'w').write('\n'.join(out)+'\n'); return wrote
w=write(sys.argv[3], False); print(f"wrote FULL: {w} -> {sys.argv[3]}")
if KEEP<1.0:
    tp=sys.argv[3].replace('.star','')+'.top%02d.star'%int(KEEP*100)
    tw=write(tp, True); print(f"wrote TOP-{int(KEEP*100)}% confidence: {tw} -> {tp}")
