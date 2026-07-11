import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, os
F="/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/results/2026_Jul_06/B2_bin1_POST/V260708.star"
rows=[]; in_fsc=False
for line in open(F):
    s=line.strip()
    if s.startswith("data_fsc"): in_fsc=True; continue
    if in_fsc and s.startswith("data_") and not s.startswith("data_fsc"): break
    if not in_fsc: continue
    if s.startswith(("loop_","_rln","#")) or s=="": continue
    p=s.split()
    if len(p)>=8:
        try: rows.append([float(x) for x in p[:8]])
        except ValueError: pass
d=np.array(rows); freq=d[:,1]; ang=d[:,2]; corrected=d[:,3]; unmasked=d[:,5]; masked=d[:,6]; prand=d[:,7]
CUT=3.0; m=ang>=CUT; fmax=1.0/CUT
def cross(fr,fsc,thr):
    for i in range(1,len(fr)):
        if fsc[i-1]>=thr and fsc[i]<thr:
            t=(fsc[i-1]-thr)/(fsc[i-1]-fsc[i]+1e-9); return 1.0/(fr[i-1]+t*(fr[i]-fr[i-1]))
    return None
r143=cross(freq[m],corrected[m],0.143); r50=cross(freq[m],corrected[m],0.5)
fig,ax=plt.subplots(figsize=(9,5.2))
ax.plot(freq[m],corrected[m],"k-",lw=2,label="Corrected"); ax.plot(freq[m],masked[m],"b-",lw=1.3,label="Masked")
ax.plot(freq[m],unmasked[m],"g-",lw=1.3,label="Unmasked"); ax.plot(freq[m],prand[m],"r-",lw=1.0,label="Phase-randomized")
ax.axhline(0.143,ls="--",c="gray",lw=0.9); ax.text(0.005,0.16,"0.143",color="gray",fontsize=8); ax.axhline(0.5,ls="--",c="gray",lw=0.7)
if r143: ax.axvline(1.0/r143,ls=":",c="k",lw=0.9); ax.text(1.0/r143,-0.13,"%.2f A"%r143,ha="center",fontsize=9)
ax.set_xlim(0,fmax); ax.set_ylim(-0.2,1.05); ax.set_xlabel("spatial frequency (1/A)"); ax.set_ylabel("Fourier Shell Correlation")
ax.set_title("FSC (cut at %g A)  --  corrected FSC=0.143 at %.2f A"%(CUT, r143 if r143 else float('nan')))
ax.grid(alpha=.3); ax.legend(loc="upper right",fontsize=9)
for out in [os.path.dirname(F)+"/V260708_fsc_cut3A.png", "/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/experiments/V260708_fsc_cut3A.png"]:
    try: plt.savefig(out,dpi=110); print("saved",out); break
    except Exception as e: print("could not save",out,e)
print("res(0.143)=%.2fA  res(0.5)=%.2fA"%(r143 or -1, r50 or -1))
