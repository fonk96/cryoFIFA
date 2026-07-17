#!/usr/bin/env python3
"""
analyze_discard.py -- does loss-ranking discard the low-SNR particles?

  python analyze_discard.py <high_stack> <keep_ratio> <loss_dir1> <loss_dir2> ...

Each loss_dir is a mix{MIX}_lrON_loss_ep{EP}/ folder holding loss_infer_rank*.txt
lines of "particle_name loss_value". high_stack (e.g. J530/J532) tags the high-SNR
source; everything else is low-SNR. For each epoch we report:
  - median loss of high vs low sub-populations (do they separate by noise floor?)
  - among the kept set (lowest keep_ratio fraction by loss), what % is high-SNR
    (50% = no discard; ->100% = loss-rank throws out the low-SNR junk)
  - recall: fraction of each sub-population that survives the cut
Writes discard_curve.csv (epoch, high_frac_kept, ...) for plotting.
"""
import sys, glob, os, re
import numpy as np
HIGH=sys.argv[1]; KEEP=float(sys.argv[2]); dirs=sys.argv[3:]

def load(d):
    names=[]; loss=[]
    for fn in glob.glob(os.path.join(d,'loss_infer_rank*.txt')):
        for ln in open(fn):
            t=ln.split()
            if len(t)<2: continue
            try: v=float(t[-1])
            except: continue
            names.append(t[0]); loss.append(v)
    return names, np.array(loss)

rows=[]
print(f"{'ep':>3} {'n_hi':>6} {'n_lo':>6} {'medL_hi':>9} {'medL_lo':>9} {'sep':>6} "
      f"{'%hi_kept':>9} {'hi_recall':>9} {'lo_recall':>9}")
for d in sorted(dirs, key=lambda x:int(re.search(r'ep(\d+)',x).group(1)) if re.search(r'ep(\d+)',x) else 0):
    m=re.search(r'ep(\d+)',d); ep=int(m.group(1)) if m else -1
    names,loss=load(d)
    if len(loss)==0: print(f"{ep:>3}  (no loss files in {d})"); continue
    hi=np.array([HIGH in n for n in names])
    order=np.argsort(loss)                    # ascending loss
    k=int(round(KEEP*len(loss)))
    kept=np.zeros(len(loss),bool); kept[order[:k]]=True   # lowest-loss keep_ratio fraction
    med_hi=float(np.median(loss[hi])) if hi.any() else float('nan')
    med_lo=float(np.median(loss[~hi])) if (~hi).any() else float('nan')
    pct_hi_kept=100*hi[kept].mean() if kept.any() else float('nan')      # composition of kept set
    hi_recall=100*kept[hi].mean() if hi.any() else float('nan')          # frac of high-SNR surviving
    lo_recall=100*kept[~hi].mean() if (~hi).any() else float('nan')      # frac of low-SNR surviving
    sep=med_lo-med_hi
    print(f"{ep:>3} {hi.sum():>6} {(~hi).sum():>6} {med_hi:>9.4f} {med_lo:>9.4f} {sep:>6.3f} "
          f"{pct_hi_kept:>8.1f}% {hi_recall:>8.1f}% {lo_recall:>8.1f}%")
    rows.append((ep,pct_hi_kept,hi_recall,lo_recall,med_hi,med_lo))

if rows:
    rows.sort()
    out=os.path.join(os.path.dirname(dirs[0].rstrip('/')) or '.','discard_curve.csv')
    with open(out,'w') as f:
        f.write("epoch,pct_high_in_kept,high_recall,low_recall,med_loss_high,med_loss_low\n")
        for r in rows: f.write("%d,%.3f,%.3f,%.3f,%.6f,%.6f\n"%r)
    print("wrote",out)
    print(f"\nNOTE: keep_ratio={KEEP}; 50% high-in-kept = no discard, higher = low-SNR discarded.")
