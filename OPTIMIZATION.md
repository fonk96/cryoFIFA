# cryoFIFA — Future Optimization Directions

Prioritised by *impact ÷ effort*. Tiers: **A** fix now (cheap, high‑value, unblocks everything
else), **B** engineering hardening, **C** method/science upside, **D** performance & scale.
Each item notes the concrete anchor in the code.

---

## Tier A — Correctness & reproducibility (do first)

**A1. Add model checkpointing.** The trainer never saves weights (no `torch.save`/`state_dict`
in `cryofifa.py`). Save `{encoder, predictor, decoder, optimizers, scaler, epoch, args}` every
`--save-interval` and support `--resume`. This is the highest‑leverage change: it enables crash
recovery, resumable long runs, **inference on new particles without retraining**, transfer/warm‑starts,
and A/B comparison of snapshots. *Effort: low. Impact: very high.*

**A2. Resolve the two CTF paths.** Keep `create_ctf` (relativistic, `-fftshift`ed), delete/rename
`compute_ctf` and the commented third copy, and **fix the `dfang` degrees‑vs‑radians mismatch**
between `get_ctf` (deg→rad) and `dataio.simulate_data` (raw rad). A wrong CTF sign/convention silently
caps resolution — this is worth a targeted unit test comparing against a RELION/cryoSPARC CTF for a
known particle. *Effort: low. Impact: high (data correctness).*

**A3. Fix or delete the silently‑broken branches.** Either repair or remove:
`new_utils.symmetrize_output` (`.real`/`.imag` slice assignment is a no‑op), the downsample path
(`new_dataloader.py:99/114`, `new_utils.py:57` don't forward args), and the `mode='2d'/'both'`
render/loss code (undefined `Encoder_2D_Predictor`/`loss_2d`). Right now flipping these flags fails or
no‑ops without warning. *Effort: low–medium. Impact: high (removes landmines).*

**A4. Ship a real FSC + gold‑standard resolution report.** The project trains half‑maps
(`half1`/`half2`) but has **no packaged FSC** — resolution is judged by eye or external tools. Add an
FSC(half1, half2) computation (the radial primitives in `utils.py` are 80% of it), write the 0.143
resolution and an FSC curve into each snapshot, and fold it into `experiment_report_generator.py`.
This turns "hundreds of runs" into something you can rank objectively. *Effort: medium. Impact: high.*

**A5. Introduce git.** Replace `V2402/V2503/V2509/V2512/Pure/` snapshot directories with a real repo
(tag the snapshots as commits). You immediately get diffs, blame, branches, and bisection — and the
`experiment_report_generator` can key experiments to commit hashes instead of dated `.sh` files.
*Effort: low. Impact: high (everything downstream gets easier).*

---

## Tier B — Engineering hardening

**B1. Tame the ~250‑flag parser.** Split `get_parser` into grouped configs (data / model / schedule /
optim / distributed) backed by `OmegaConf`/`dataclass` YAML (the env already has `omegaconf`, `yacs`).
Snapshot the resolved config into each `render_path` for provenance. Delete flags that only gate dead
branches. This is the biggest single maintainability win after git. *Effort: medium. Impact: high.*

**B2. De‑duplicate the Fourier/CTF/mask/rotation code.** Collapse to one implementation each:
`ft2/ift2/ft3/ift3` (currently in both `utils.py` and `fourier_utils.py` with divergent norm/upcast
behaviour and import‑order‑dependent shadowing), `which_half_space` (×4), `Symmetrizer`,
`image_duplicator`, `slicer`, `sigproc`↔`torch_sigproc`, and the 3+ 6D‑orthonormalisation routines.
Decide whether `fourier.py`'s orthonormal pyFFTW stack is still needed; if not, drop it. *Effort:
medium. Impact: high (eliminates a whole class of silent convention bugs).*

**B3. Add an LR schedule.** Fixed Adam LR wastes the strong coarse‑to‑fine curriculum. A warmup +
cosine/step decay tied to the frequency‑marching schedule (higher LR at low res, decay as high‑freq
shells open) should improve both convergence speed and final resolution. *Effort: low. Impact: medium.*

**B4. Quarantine dead code.** Move to `archive/`: `parnet.py`, `backup_build_encoder.py`,
`cryosparc_simulator.py` (fragment), `fts.py`, `ft2.py`, and the `half3d_new.py` legacy trainer (which
won't even run against the current `Fullnet` API). Remove the unused `from swd import swd`. This shrinks
the surface a newcomer must understand by ~30%. *Effort: low. Impact: medium.*

**B5. Real logging + tests.** Replace carriage‑return `print` progress with TensorBoard/W&B (loss per
shell, pose‑uniformity, FSC, GPU mem). Stop silencing all warnings globally. Add physics unit tests:
FFT round‑trip, CTF vs reference, `phase_shift_ft` vs a known translation, projection–slice consistency
on a synthetic volume. *Effort: medium. Impact: medium–high (catches regressions the snapshots can't).*

---

## Tier C — Method / science upside

**C1. Replace best‑of‑N symmetric pose loss with a marginalised / branch‑and‑bound search.**
The `sym_factor` multi‑hypothesis min‑loss is effective but its cost scales linearly with the number of
in‑plane copies and it only searches in‑plane. Consider a small **SO(3) importance‑sampling or
hierarchical pose search** (cryoDRGN‑style) over a few epochs, or a probabilistic (soft‑min /
log‑sum‑exp) pose posterior, to improve ab‑initio robustness on hard, low‑SNR or preferred‑orientation
sets. *Effort: high. Impact: high (this is the crux of ab‑initio quality).*

**C2. Validate & extend the heterogeneity model.** The continuous‑latent FiLM decoder and the discrete
`cryoFIFA_Multiclass` are two separate scripts. Unify them, and benchmark the continuous latent against
cryoDRGN on a standard heterogeneous set (spliceosome, ribosome assembly) with quantitative metrics
(per‑state FSC, latent‑space UMAP vs known classes). Add latent regularisation (KL/β‑VAE — partly
scaffolded in `half3d_new`) to get a smooth, traversable conformational space. *Effort: high. Impact:
high (this is the differentiating capability).*

**C3. Re‑enable and test Hermitian symmetrisation of the decoder output.** `Fullnet.use_symmetrizer`
and `symmetrize_output` are plumbed but inert/broken (see A3). Enforcing `F(-k)=conj(F(k))` on the
implicit field guarantees a real reconstruction and can reduce the "ring" artifacts the code comments
complain about, potentially removing the need for some ad‑hoc 3D masking. *Effort: medium. Impact:
medium.*

**C4. Point‑group symmetry as a first‑class option.** Many test specimens (apoferritin O, ribosome C1,
GPCRs C1, but also Cn/Dn cases) have known symmetry. The `image_duplicator`/`sym_factor` machinery is
about the *loss*, not the *volume*. Adding true molecular‑symmetry averaging in the Fourier decoder
(symmetry‑expanded coordinate sampling) would boost SNR and resolution for symmetric particles.
*Effort: medium. Impact: medium–high on symmetric sets.*

**C5. Learned or amortised CTF/pose refinement.** Once poses converge, a light per‑particle pose+CTF
refinement stage (à la RELION/cryoSPARC polishing) on top of the amortised encoder could push
resolution. The pieces (per‑particle CTF params, `phase_shift_ft`) already exist. *Effort: medium.*

---

## Tier D — Performance & scale

**D1. Cache PDO steerable kernels & remove hardcoded CUDA.** `PDO_models` recomputes steering
kernels every forward and hardcodes `.to('cuda')`. Cache kernels as buffers and take `device` from the
input — this both speeds up the `pdo` backbone and makes CPU/MPS debugging possible. *Effort: low.*

**D2. Vectorise the volume render.** The snapshot render decodes the volume in a Python
`for slicei in range(imgsize)` loop. Batch slices (chunked by available memory) to cut render time,
which matters at box 256–360. *Effort: low.*

**D3. Vectorise pose‑star writing.** The `--save_pose` path concatenates strings and calls
`.cpu().numpy()` per particle in a Python loop. Build a pandas frame and write once (or use the
`starfile` library). *Effort: low.*

**D4. Data‑loading throughput.** LMDB `float16` baking is good; verify `num_workers`/`pin_memory` are
tuned, and consider storing **pre‑FFT'd** particles (the loss is computed on `ft2(y)` every step —
precomputing and caching the particle FT trades disk for repeated GPU FFTs). Profile whether the
CPU→GPU `ft2` per batch is a bottleneck at large boxes. *Effort: medium.*

**D5. Revisit mixed precision.** `float16` + manual complex handling is fragile (the `utils.ft2`
upcasts to fp32 to avoid `inf`). Evaluate `bfloat16` for the encoder and keep FFT/loss in fp32, or
adopt `torch.autocast` policies per‑module, for more stable high‑resolution training. *Effort: medium.*

---

## Suggested sequence

1. **A5 (git) + A1 (checkpoint) + A2 (CTF)** — a weekend's work that de‑risks everything else.
2. **A3 + A4 (kill silent branches, add FSC)** — makes runs trustworthy and rankable.
3. **B1 + B2 + B4 (config, de‑dup, quarantine)** — pays down the maintainability debt.
4. Then invest in **C1 / C2** (pose search + heterogeneity), which is where the scientific
   differentiation actually lives, with **D** perf items folded in opportunistically.

The through‑line: the *science* is already competitive; the *engineering discipline* (versioning,
checkpoints, config, de‑duplication, objective FSC metrics) is what currently limits how fast the
method can be validated, trusted, and improved.
