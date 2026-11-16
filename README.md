# cryoFIFA — Project Documentation

> Fourier‑domain **I**mplicit‑**F**unction **A**uto‑encoder for ab‑initio / heterogeneous
> single‑particle **cryo**‑EM 3D reconstruction.
> (The `AE` in the legacy trainer name `kryoAE_trunk.py` confirms the autoencoder framing;
> "FIFA" is the author's project codename — treat the exact expansion as informal.
> The current entry point is `src/cryofifa_ddpfix.py`; older trainers live in `archive/`.)

Source analysed: `/sddn/kongfang_work/cryoFIFA/scripts/` (~11 k lines of Python, ~40 files),
plus dated full‑directory snapshots `V2402/ V2503/ V2509/ V2512/ Pure/` (the project's
poor‑man's version control — there is **no git repo**). A cleaned‑up copy of the code has been
reorganized into `src/ tools/ tests/ configs/ experiments/ archive/` here — see **STRUCTURE.md**.

This directory contains:

| File | Contents |
|------|----------|
| **README.md** (this file) | What the project does, the method, the architecture, how a run works, and the experimental scope. |
| **ASSESSMENT.md** | Scientific judgment: what's genius, why results plateau near 4 Å, and where the resolution headroom is. |
| **CODE_REVIEW.md** | Detailed, file‑by‑file code review: structure, design choices, and a catalogue of confirmed bugs, duplication, dead code, and hardcoded values. |
| **OPTIMIZATION.md** | Prioritised future optimization directions (correctness, engineering, method/science, performance). |
| **STRUCTURE.md** | The reorganized directory layout and how to run from the copy. |

---

## 1. What this project does

cryoFIFA reconstructs a **3D electron‑density map** (and, optionally, a *continuum* or
*discrete set* of conformational states) directly from a stack of noisy 2D cryo‑EM
particle images — **without requiring pre‑determined particle orientations** (ab‑initio).
It is a research reimplementation/extension in the same family as **cryoDRGN**,
**cryoAI**, and **cryoSPARC's ab‑initio**, with two distinctive commitments:

1. **The 3D volume is an implicit neural field in Fourier space.** Instead of a voxel grid,
   a coordinate MLP (SIREN) maps a 3D frequency coordinate `(kx, ky, kz)` → a **complex
   Fourier value** `(real, imag)`. The volume is never materialised during training.

2. **The whole forward model lives in Fourier space** and is differentiable end‑to‑end,
   exploiting the **projection–slice theorem**: the 2D Fourier transform of a projection
   equals a **central slice** through the 3D Fourier volume. Rendering a predicted image is
   therefore just "rotate a 2D coordinate plane, sample the implicit Fourier field, apply
   CTF and a phase shift."

The encoder (a 2D CNN) *amortises* pose inference: it looks at each particle and predicts
its orientation, translation, and (optionally) a latent heterogeneity code. This is the
"auto‑encoder" structure — encoder → pose/latent → physics‑based Fourier decoder → image →
reconstruction loss against the input.

### Inputs and outputs

- **Input:** RELION/cryoSPARC `.star` metadata + `.mrcs` particle stacks. For speed these are
  pre‑baked into an **LMDB** blob store (raw `float16` buffers keyed by integer index). CTF
  parameters (defocus U/V/angle, voltage, Cs, amplitude contrast, pixel size) are read from the
  star optics + per‑particle rows.
- **Output:** reconstructed 3D maps written periodically as MRC volumes
  (`rec_3d_<epoch>_<minutes>min.mrc`), and — when `--save_pose` is set — a `.star` file of the
  **predicted per‑particle poses** (Euler angles + raw 6D rotation), so results can be fed back
  into RELION/cryoSPARC. **Model weights are *not* checkpointed** (see CODE_REVIEW / OPTIMIZATION).


---

## 2. How a training run works

Runs are launched from a shell script under `experiments/` that `source`s `paths`
(dataset name → `.star` path map) and calls `torchrun --nproc_per_node 8 ../src/cryofifa_ddpfix.py …`
with a long flag string. A representative invocation (β2‑adrenergic, bin1, half‑map):

```bash
torchrun --nproc_per_node 8 ../src/cryofifa_ddpfix.py \
  --particles_meta ${B2_act_half2} --lmdb_path /data/lmdb/B2_act_half2 \
  --imgsize 228 --data_path /data/cryoFIFA_data/ \
  --encoder_backbone resnet --encoder_basename resnet18 \
  --decoder_layers 2 --decoder_hidden_dim 256 --activation SIREN --w0 48 --w 32 \
  --fourier --rotation6d --norm6d --sym --sym_factor 8 \
  --start_resolution 40 --end_resolution 4 --mask_steps 64 --mask_epoch 3 \
  --num-epochs 50 --minibatch-size 32 --batch_size 32 --float16 --preprocess \
  --add_distrubance --distrubance_ratio 0.5 --distrubance_epoch 70 --uniform \
  --world_size 8 --free_gpu_memory 8 --render_path results/<date>/<run>/
```

The loop (see `cryofifa_ddpfix.py`): for each epoch it selects the current resolution shell,
optionally shrinks the minibatch if free GPU memory drops below `--free_gpu_memory`,
runs the encoder→decoder→Fourier forward model under AMP, accumulates gradients up to
`--batch_size`, steps the (separate) Adam optimizers, and every `--save-interval` epochs
decodes the **full** 3D Fourier volume slice‑by‑slice, inverse‑FFTs it, masks/normalises,
and writes an MRC. Half‑maps (`half1`/`half2`) are trained independently so an external
FSC can estimate resolution.

**Two lines of prior art it most resembles:** cryoDRGN (implicit volume + latent
heterogeneity) and cryoAI (amortised pose encoder + Fourier‑slice + symmetric loss).
cryoFIFA's novelty is the **envelope×modulant Fourier decoder**, the aggressive
frequency‑marching + noise‑whitening schedule, and the engineering for large boxes
(LMDB, dynamic minibatching, `float16` DDP).

---

See **CODE_REVIEW.md** for the file‑by‑file assessment and defect catalogue, and
**OPTIMIZATION.md** for where to take it next.
