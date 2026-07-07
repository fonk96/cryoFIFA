# cryoFIFA — Scientific Assessment

*How I judge the approach, what is genuinely clever about it, why it currently plateaus near
4 Å, and where the real resolution headroom is.*

This is an opinionated critique of the **method**, not the code hygiene (that's in CODE_REVIEW.md).
It assumes the stated result: **most good reconstructions land at mid‑to‑near 4 Å.**

---

## 1. Where "near 4 Å" actually sits

Resolution is specimen‑dependent, so the number means different things across your benchmark set:

- On a **ribosome / apoferritin / β‑gal** (large, rigid, high‑contrast), 4 Å is a *modest* result —
  RELION/cryoSPARC reach 2–3 Å routinely on the same particles, and apoferritin goes below 2 Å.
- On a **small GPCR like β2‑adrenergic (~50 kDa), pfCRT, or a membrane transporter**, 4 Å **ab‑initio
  and pose‑free** is genuinely respectable — these are the specimens where amortized neural methods
  earn their keep, because getting *any* consistent ab‑initio solution is hard.

So the honest framing: cryoFIFA is **competitive with the cryoAI / cryoFIRE class** of amortized,
pose‑free, Fourier‑space neural reconstructors, and its resolution ceiling is in the same place theirs
is. It is **not yet competitive with iterative refinement** (RELION/cryoSPARC) on easy specimens.
That gap is not a failure of the representation — it is the well‑known cost of **amortized,
single‑shot pose inference**, discussed in §3.

The 4 Å band is exactly the regime where the *representation* is no longer the bottleneck and the
*pose accuracy* is. That is the single most important thing to internalize about this result.

---

## 2. What is genius here

Three ideas are genuinely good, and one of them I think is novel enough to be the paper's headline.

### 2a. The envelope × modulant Fourier decoder — the standout idea

`Fullnet` represents the 3D structure factor as

```
    F(k) = exp( g_env(k) ) · h_mod(k)
```

with two SIREN networks. The motivation in the code (an exponential real‑space density prior
`f(x)=λe^{−λx}` has Fourier transform `λ/(λ−it)`) is a real, principled inductive bias. What it buys:

- **It factorizes "amplitude falloff" from "structure."** `exp(g_env)` is a smooth, strictly‑positive
  envelope — physically the analogue of a B‑factor / solvent‑and‑disorder decay — while `h_mod`
  carries the oscillatory structure factor (where the atoms are). Learning these separately is a
  much better‑conditioned problem than asking one network to produce a complex value that must both
  decay over three orders of magnitude *and* oscillate.
- **It is a soft spectral prior that stabilizes early training.** During the low‑resolution phase of
  frequency marching, the envelope can be fit almost immediately from a handful of shells, freeing the
  modulant to focus on phase/structure. This is a large part of *why the ab‑initio problem converges
  at all* here.

I have not seen this exact decomposition elsewhere (cryoDRGN uses a positional‑encoding MLP in real
space; cryoAI uses a plain SIREN in Fourier space). It is the most defensible "genius" claim.

### 2b. SIREN in Fourier space + direct central‑slice sampling

Putting a **sinusoidal‑activation** network in **Fourier** space is a natural fit: structure factors
are oscillatory, and SIREN's `w0` gives you an explicit knob on spectral bandwidth (which pairs
beautifully with frequency marching — you're literally growing the bandwidth of both the mask and the
network together). And because the volume *is* the Fourier field, image formation is a **direct
grid‑sample of a rotated 2D plane** (projection–slice theorem) — no ray‑marching, no line integrals,
one differentiable slice per particle. This is the cryoAI insight, executed with a stronger decoder.

### 2c. The training curriculum is the real workhorse

The convergence of a pose‑free method is 90% curriculum, and this one is well‑stocked and mutually
reinforcing:

- **Frequency marching** (`reconstruct_fourier_masks`): expand the active resolution 30 Å → 4 Å,
  so early gradients are noise‑robust and pose search happens on smooth low‑res landscapes.
- **Multi‑hypothesis symmetric loss** (`image_duplicator` + `symmetric_complex_loss`'s min‑over‑paths):
  a differentiable branch‑and‑select that fights pose multimodality.
- **Pose disturbance annealing** (`add_distrubance` + `cosine_decay`): injected rotation noise early
  to escape bad basins, decayed to zero — simulated‑annealing for poses.
- **Pose‑uniformity regularizer** (`uniform_s2_moment_loss`): matches the predicted rotation
  distribution's moments to uniform‑on‑S², directly countering the orientation‑collapse failure mode
  that kills naive amortized encoders.
- **Shell/dose whitening** (`make_shell_weights_midband`, `--DW`): keeps mid‑band frequencies from
  being drowned by low‑frequency power or high‑frequency noise.

Wrapped in serious systems engineering (LMDB `float16` baking, DDP + AMP, gradient accumulation,
`--reduce_input` sparse decoding, and **live‑GPU‑memory‑driven dynamic minibatch resizing**) that
makes 200–360‑px boxes tractable on 8 GPUs. The curriculum + engineering is why it *works*, even if
2a is why it's *interesting*.

---

## 3. Why it plateaus near 4 Å — the diagnosis

The ceiling is almost certainly **pose accuracy**, with three contributing causes and one likely
schedule artifact. In order of suspected impact:

### 3a. Amortized single‑shot pose is the fundamental cap *(primary)*

The encoder predicts **one** 6‑D rotation per image in a single forward pass. High‑resolution signal
lives in the highest frequencies, and those are exactly the ones destroyed by small pose errors: an
angular error `Δθ` blurs detail beyond roughly `d ≈ (box/2)·Δθ` — so **even ~1–2° of systematic pose
error caps you right around 4 Å** on a typical box. An amortized encoder, however good, produces a
*point estimate*; it cannot self‑consistently marginalize or iteratively polish the pose the way
RELION's expectation‑maximization or cryoDRGN's hierarchical pose search does. This is the same wall
cryoAI and cryoFIRE hit, and for the same reason.

### 3b. The multi‑hypothesis search is only *in‑plane* *(secondary)*

`image_duplicator` generates copies by `rot90`/flip/2D‑rotate, i.e. it explores **in‑plane** rotation
multimodality. The genuinely hard ambiguity in ab‑initio is **out‑of‑plane** (viewing direction) and
the handedness/mirror ambiguity. Those are handled only by the encoder's point estimate plus
disturbance noise — no explicit SO(3) search. So the cleverest anti‑local‑minimum mechanism in the
code doesn't cover the axis that matters most for the last Å.

### 3c. The envelope may be over‑smoothing high frequencies *(contributory)*

`exp(g_env)` enforces a smooth, near‑monotonic amplitude spectrum. That is a great prior early, but at
high resolution real structure factors are not smooth, and a too‑soft envelope will **suppress genuine
high‑frequency amplitude** — trading the early‑convergence win for a late‑resolution cap. The `exp`
is also numerically twitchy in `float16` (the code carries clamp comments), which discourages pushing
it hard. The inductive bias that helps you reach 6 Å may be part of what stops you past 4 Å.

### 3d. The schedule may *literally* stop at 4 Å *(check this first — it's free)*

The production runs use `--end_resolution 4`. Frequency marching admits shells only up to
`end_resolution` (clamped to Nyquist `2·apix`). If the loss never sees frequencies past 4 Å, **you
cannot reconstruct past 4 Å by construction** — independent of everything above. Before concluding
the method is capped, push `--end_resolution` to 3 (or `2·apix`) and see whether it holds. If it does,
the cause is 3a–3c; if it degrades, the schedule/whitening past 4 Å needs work. Either way you learn
something for one flag's worth of effort.

### Also worth ruling out

- **No gold‑standard FSC in the loop.** Half‑maps are trained separately (good), but resolution is
  only as trustworthy as the *external* FSC, and there's no built‑in overfitting guard beyond the mask.
- **CTF is standard 2D** — no Ewald‑sphere curvature, no beam‑tilt / higher‑order aberration, no
  per‑particle B‑factor. Fine to ~3 Å; part of the last fraction of an Å below that.

---

## 4. Where it can move further — ranked by expected resolution gain

**Tier 1 — break the pose ceiling (this is the whole ballgame).**

1. **Add a pose *refinement* stage on top of the amortized encoder.** You already have a fully
   differentiable forward model (`phase_shift_ft`, per‑particle CTF, central‑slice). After the encoder
   converges, do a short per‑particle **gradient‑based pose+shift+CTF polish** (a handful of steps),
   or a **local exhaustive SO(3) search** around the encoder's estimate. This is the single most
   likely thing to move you from 4 Å toward 3 Å, and most of the machinery exists.

2. **Make the pose search cover SO(3), not just in‑plane.** Extend the multi‑hypothesis mechanism from
   in‑plane image copies to a small set of **viewing‑direction hypotheses** (importance sampling or a
   coarse S² grid), and replace the hard `min` with a **soft‑min / log‑sum‑exp posterior** so
   gradients flow to *all* plausible poses early, collapsing to one as marching proceeds. This is the
   cryoDRGN‑style marginalization idea, grafted onto your amortized encoder.

3. **Two‑stage: amortized global → auto‑decoder local.** Use the fast encoder to get a consistent
   global solution, then switch to a **per‑particle latent+pose optimization** (DeepSDF‑style; the
   `auto_decoder` mode already exists in the legacy `half3d_new.py`) for the final push, where
   per‑particle freedom matters more than amortization speed.

**Tier 2 — let the representation express high‑res detail.**

4. **Relax the envelope at high frequency.** Try `softplus` instead of `exp`, or add a **learnable
   per‑shell amplitude correction / B‑factor sharpening** term so the envelope prior helps early but
   doesn't cap late. Ablate: does removing/relaxing the envelope past a given shell raise the FSC?

5. **Push `--end_resolution` and fix the whitening past 4 Å** (see 3d). Cheap, do it first.

**Tier 3 — the last fraction of an Å.**

6. **Better noise model / likelihood.** Replace the heuristic shell weighting with a proper
   **per‑shell (ideally per‑particle) Gaussian likelihood** with estimated variance — this is both a
   principled loss and an automatic whitening, and it's what the classical methods do.

7. **Aberration‑aware CTF** (Ewald curvature, beam tilt, trefoil) for large boxes / sub‑3 Å targets.

8. **Re‑enable Hermitian symmetrization of the decoder output** (`Fullnet.use_symmetrizer` /
   `symmetrize_output` are plumbed but inert/broken). Guaranteeing `F(−k)=conj(F(k))` yields an
   exactly‑real volume and can remove the ring artifacts the code comments complain about — possibly
   letting you drop some ad‑hoc 3D masking that itself costs resolution.

**Tier 4 — capability, not resolution.**

9. **Upgrade heterogeneity.** The single global latent modulating a global SIREN is coarse for
   continuous motion; a **deformation‑field / flow decoder (3DFlex‑like)** or hierarchical latents,
   plus a real β‑VAE/KL for a smooth traversable landscape, would make the continuous‑heterogeneity
   story competitive with cryoDRGN. Unify this with the discrete `Multiclass` path.

**Validation infrastructure that makes all of the above measurable.**

10. **Put gold‑standard FSC in the training loop** and log it per snapshot (OPTIMIZATION.md · A4).
    Right now you can't tell an ablation win from noise without leaving the pipeline. This is the
    cheapest high‑value change and it gates honest progress on 1–9.

---

## 5. Bottom line

The **representation and the curriculum are the intellectual contribution**, and the
envelope×modulant Fourier decoder is a genuinely elegant, defensible idea. Near‑4 Å from a
pose‑free amortized method — especially on small membrane proteins — is a solid, publishable‑class
result in the cryoAI/cryoFIRE lineage.

The **plateau is a pose‑accuracy plateau, not a representation plateau.** The path past 4 Å runs
through *pose*: add refinement/marginalization over full SO(3) (Tier 1), then let the decoder express
high frequencies (Tier 2), and instrument everything with in‑loop FSC so you can actually see the
gains. Do the one‑flag `end_resolution` check first — it tells you immediately whether you're fighting
the method or just the schedule.
