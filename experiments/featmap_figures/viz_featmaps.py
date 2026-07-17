import numpy as np, glob, os, sys, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.family':'DejaVu Sans','savefig.bbox':'tight'})
SP='/tmp/claude-1000/-sddn-kongfang-work-cryoFIFA-V260706-cryoFIFA/5b6a4386-0d4c-46ae-a59a-407ec3a82e9b/scratchpad/'
FM=SP+'featmaps/'
files=sorted(glob.glob(FM+'ep*.npz'))
assert files, 'no ep*.npz found'
snaps=[np.load(f) for f in files]
epochs=[int(s['epoch']) for s in snaps]
STAGES=['input','stem','layer1','layer2','layer3','layer4']
STLAB={'input':'input','stem':'stem (conv1)','layer1':'layer1','layer2':'layer2','layer3':'layer3','layer4':'layer4'}
n=snaps[0]['input'].shape[0]
print('epochs',epochs,'particles',n)

def mag(t):                     # [C,H,W] -> [H,W] activation magnitude (L2 over channels)
    return np.sqrt((t.astype('float32')**2).sum(0))
def norm(m):
    m=m-m.min(); return m/(m.max()+1e-9)

# ---- per-particle: rows=epoch, cols=stage (activation magnitude maps) ----
NP=min(n,6)                     # first 6 particles as separate figures
for p in range(NP):
    R,C=len(epochs),len(STAGES)
    fig,ax=plt.subplots(R,C,figsize=(C*1.85,R*1.85))
    ax=np.atleast_2d(ax)
    for r,s in enumerate(snaps):
        for c,st in enumerate(STAGES):
            a=ax[r,c]; a.imshow(norm(mag(s[st][p])),cmap='turbo',interpolation='bilinear')
            a.set_xticks([]); a.set_yticks([])
            if r==0: a.set_title(STLAB[st],fontsize=11)
            if c==0: a.set_ylabel(f'epoch {s["epoch"]}',fontsize=11)
    fig.suptitle(f'Encoder feature-map evolution — particle {p}  (activation magnitude, per-cell normalized)',fontsize=12,y=1.005)
    plt.tight_layout()
    fig.savefig(SP+f'fig_featmap_particle{p}.png',dpi=140); plt.close(fig)
print(f'saved fig_featmap_particle0..{NP-1}.png')

# ---- overview: 1 particle per row, cols=stage at the FINAL epoch ----
last=snaps[-1]
fig,ax=plt.subplots(n,len(STAGES),figsize=(len(STAGES)*1.7,n*1.7)); ax=np.atleast_2d(ax)
for p in range(n):
    for c,st in enumerate(STAGES):
        a=ax[p,c]; a.imshow(norm(mag(last[st][p])),cmap='turbo',interpolation='bilinear')
        a.set_xticks([]); a.set_yticks([])
        if p==0: a.set_title(STLAB[st],fontsize=11)
        if c==0: a.set_ylabel(f'ptcl {p}',fontsize=10)
fig.suptitle(f'Encoder feature maps for 10 particles (epoch {last["epoch"]})',fontsize=12,y=1.004)
plt.tight_layout(); fig.savefig(SP+'fig_featmap_allparticles_final.png',dpi=140); plt.close(fig)
print('saved fig_featmap_allparticles_final.png')

# ---- individual learned channels: particle 0, layer1, evolution first->last epoch ----
for st in ['layer1','layer2']:
    t=last[st][0].astype('float32'); Cn=min(t.shape[0],64); g=int(np.ceil(np.sqrt(Cn)))
    fig,ax=plt.subplots(g,g,figsize=(g*1.0,g*1.0))
    for i in range(g*g):
        a=ax[i//g,i%g]; a.axis('off')
        if i<Cn: a.imshow(norm(t[i]),cmap='turbo',interpolation='bilinear')
    fig.suptitle(f'{st} individual channels — particle 0, epoch {last["epoch"]} (first {Cn})',fontsize=12)
    plt.tight_layout(); fig.savefig(SP+f'fig_featmap_channels_{st}.png',dpi=140); plt.close(fig)
print('saved fig_featmap_channels_layer1/2.png')
