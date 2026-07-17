import numpy as np, os, sys, matplotlib
matplotlib.use('Agg')
# Arial + EDITABLE TEXT so fonts can be adjusted in Illustrator/Inkscape:
#  svg.fonttype='none' keeps <text> elements referencing Arial by name (not outlined);
#  pdf/ps.fonttype=42 embeds TrueType (still selectable/editable text).
matplotlib.rcParams.update({
    'svg.fonttype':'none', 'pdf.fonttype':42, 'ps.fonttype':42,
    'font.family':'Arial',   # written verbatim into the SVG (svg.fonttype='none') so your editor uses Arial
    'axes.linewidth':0.8, 'xtick.direction':'out', 'ytick.direction':'out',
    'savefig.bbox':'tight', 'font.size':11})
import matplotlib.pyplot as plt
PLOTS="/sddn/kongfang_work/cryoFIFA/V260706/cryoFIFA/plots/"
OUT="/tmp/claude-1000/-sddn-kongfang-work-cryoFIFA-V260706-cryoFIFA/5b6a4386-0d4c-46ae-a59a-407ec3a82e9b/scratchpad/fsc_figs/"
os.makedirs(OUT,exist_ok=True)
# (box, apix) per file -- fill in as known
CFG={
 'B2_FSC.txt':(228,1.0825),
 'EMPIR10180_FSC.txt':(128,4.2475),
 'TRPM8_FSC.txt':(128,2.6475),
 'RAG_FSC.txt':(128,1.8450),
}
def nice_res_ticks(sf_max):
    # resolution r sits at x = 1/r; keep only ticks spaced enough on the (linear) freq axis
    cands=[50,30,20,15,12,10,8,7,6,5,4,3.5,3,2.5,2.2,2]
    out=[]; lastx=-1e9; gap=sf_max*0.085
    for r in cands:
        x=1.0/r
        if x>sf_max*1.001: continue
        if x-lastx>=gap: out.append(r); lastx=x
    return out
for fn,(box,apix) in CFG.items():
    if box is None: print(f"[skip] {fn}: need (box,apix)"); continue
    d=np.loadtxt(PLOTS+fn,skiprows=1); wave=d[:,0]; fsc=d[:,1]
    sf=wave/(box*apix)                     # spatial frequency (1/A)
    # 0.143 crossing
    cr=np.where(fsc<0.143)[0]; res143=(box*apix/wave[cr[0]]) if len(cr) else np.nan
    fig,ax=plt.subplots(figsize=(4.3,3.2))
    ax.plot(sf,fsc,'-',color='#111111',lw=1.7)
    ax.axhline(0.143,ls='--',color='0.55',lw=0.9)
    ax.set_xlim(0,sf.max()); ax.set_ylim(-0.05,1.03)
    ax.set_ylabel('Fourier shell correlation'); ax.set_xlabel('Resolution (Å)')
    rt=nice_res_ticks(sf.max())
    ax.set_xticks([1.0/r for r in rt]); ax.set_xticklabels([f'{r:g}' for r in rt])
    ax.text(0.97,0.93,f'{res143:.2f} Å',transform=ax.transAxes,ha='right',va='top',fontsize=10)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    base=fn.replace('_FSC.txt','').replace('.txt','')
    for ext in ('svg','pdf','png'): fig.savefig(OUT+f'fsc_{base}.{ext}',dpi=300)
    plt.close(); print(f"[ok] {fn}: FSC=0.143 at {res143:.2f} A  -> fsc_{base}.svg/pdf/png")
