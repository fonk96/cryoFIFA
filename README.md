# cryoFIFA — Project Documentation

> Fourier‑domain **I**mplicit‑**F**unction **A**uto‑encoder for ab‑initio / heterogeneous
> single‑particle **cryo**‑EM 3D reconstruction.
> (The `AE` in the current trainer name `kryoAE_trunk.py` confirms the autoencoder framing;
> "FIFA" is the author's project codename — treat the exact expansion as informal.)

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

### Experimental scope

The `data*/` folders hold ~90 `.star` files spanning a broad benchmark suite of real and
simulated specimens: **ribosome** (the dominant test case, 30 kDa–MDa), the **β2‑adrenergic
receptor** and other GPCRs, **pfCRT**, **TRPV1 / TRPM8** ion channels, **spliceosome**,
**γ‑secretase**, **apoferritin**, **β‑galactosidase**, **RAG**, **GAT1**, **megabody**,
**streptavidin**, **glycan**. Box sizes range 64–360 px, with simulated SNR sweeps
(0.03–0.5). The run‑script history (`oldscripts/run_2025_*.sh … run_2026_0104.sh`) records
**hundreds of experiments** — this is an actively‑iterated research codebase, not a released tool.

---

## 2. Method — the forward model, step by step

For one particle image `y` (real space) with known CTF:

```
                         ┌─────────────────────────── ENCODER (amortised inference) ──────────────────────────┐
  y (DxD real image) ──► preprocess (multi‑band       Feature_Extractor            Encode_Predictor
                          low‑pass → N channels)  ──►  (ResNet18 / ConvNeXt / …) ─► 256‑d ─► [ rot6d(6) | latent | shift(2) ]
                                                                                                     │        │        │
                                                                                                     ▼        │        ▼
        x_coord_plane (DxD, z=0 grid) ── rotation_6d_to_matrix(rot6d) ──► rotated central‑slice grid │   translation
                                                                                     │               │        │
                                                                                     ▼               ▼        │
                                          ┌──────────────── DECODER (implicit Fourier volume) ───────┘        │
                                          │  Fullnet(coords, latent) = exp(net_envelope) · net_modulant       │
                                          │     each = SIREN MLP, FiLM‑modulated by Modulator(latent)         │
                                          └──────────────────────────┬──────────────────────────────────────┘│
                                                                     ▼                                        │
                              predicted complex Fourier slice ── phase_shift_ft(·, translation) ◄─────────────┘
                                                                     │
                                                    × CTF  ── × progressive low‑pass mask ── ÷ noise/dose weights
                                                                     ▼
                             loss = symmetric_complex_loss( pred_ft , FT(y) )   (per‑pixel complex L2 in Fourier space)
```

Key mechanisms that make ab‑initio training converge:

- **Frequency marching (coarse‑to‑fine).** A schedule of concentric Fourier shell masks
  (`reconstruct_fourier_masks`) grows the active resolution from ~30 Å to ~4–6 Å over the
  epochs. Early on the network only has to fit low frequencies (robust to noise, few local
  minima); resolution is admitted gradually.

- **Symmetric multi‑hypothesis pose loss.** Each image is duplicated `sym_factor` times
  (`image_duplicator`, in‑plane rotated copies); the encoder predicts a pose per copy; the loss
  keeps the **minimum over the copies** (`symmetric_complex_loss` returns `min_distances,
  activated_paths`). This is a differentiable branch‑and‑select that mitigates the pose
  multi‑modality that sinks single‑hypothesis amortised methods.

- **Pose disturbance annealing.** Gaussian/uniform noise is injected into the predicted 6D
  rotation during early epochs (`add_distrubance`, decayed by `cosine_decay`) to escape bad
  initial pose basins.

- **Pose‑uniformity regularizer.** `uniform_s2_moment_loss` matches the 1st/2nd moments of the
  predicted rotation axes to a uniform distribution on S², discouraging orientation collapse.

- **Noise whitening / dose weighting.** Per‑shell weights (`make_shell_weights_midband`,
  `--DW`) reweight the Fourier loss so mid‑band frequencies are not swamped by low‑frequency
  power or high‑frequency noise.

- **Heterogeneity.** A latent code from the encoder FiLM‑modulates the decoder SIREN
  (`Modulator`). `cryoFIFA_Multiclass.py` instead uses a **discrete** latent
  (softmax→argmax→one‑hot) to reconstruct a fixed number of distinct classes.

---

## 3. Architecture components

| Role | Class / file | Notes |
|------|--------------|-------|
| **Encoder backbone** | `Feature_Extractor` (`merged_models.py`) | Pluggable: `resnet` (default resnet18), `convnext`, `swin`, `mobile`, `efficient`, `maxvit`, `pdo`, `mlp`. First conv rewired to 1‑or‑N low‑pass channels; classifier → identity → `Naive_MLP → 256‑d`. |
| **Rotation‑equivariant backbone** | `PDO_eConvs` (`PDO_models.py`) | Steerable p8 group‑CNN (Partial Differential Operator equiv‑convs) selectable as `backbone='pdo'`. |
| **Pose/latent head** | `Encode_Predictor` (`merged_models.py`) | Three MLP heads → `[rot6d(6), latent, shift(2)]`. |
| **Decoder (3D Fourier field)** | `Fullnet` → `Homo` → `SirenNet` + `Modulator` (`merged_models.py`) | Output `exp(net_envelope) · net_modulant`: a positive envelope × oscillating modulant, motivated by the exponential‑density ↔ `λ/(λ−it)` Fourier prior. Latent enters via FiLM modulation. |
| **Pose representation** | 6D continuous rotation (`rotation_6d_to_matrix`, pytorch3d) | Legacy trainer also has quaternion/AF2‑quaternion/Euler/2D options. |
| **FFT / CTF** | `fourier_utils.py` (`ft2/ift2/ft3/ift3`, `create_ctf`, `phase_shift_ft`) | Centered convention `fftshift∘fftn∘ifftshift`; relativistic CTF in cycles/Å. |
| **Data** | `new_dataloader.py` (`LmdbDataSet`, `StarDataSet`), `untiled.py` (`Dataloader`, star/MRC IO + LMDB baking) | Hand‑rolled star parser; LMDB is the production path. |
| **Training toolbox** | `utils.py` (1559 lines), `new_utils.py` | ~250 argparse flags, masks, losses, coordinate grids, frequency schedule, pose‑uniformity loss, CTF. |
| **Trainers** | `cryofifa.py` / `kryoAE_trunk.py` (current), `cryofifa_2026.py`, `cryofifa_back.py`, `half3d_new.py` (legacy), `cryoFIFA_Multiclass.py` (discrete het.) | torchrun + NCCL DDP, AMP `float16`, per‑model Adam, gradient accumulation, dynamic minibatch sizing. |
| **Reporting / conversion / interpretability** | `experiment_report_generator.py` (PDF of tri‑view slices), `kcs2star.py` (cryoSPARC→star), `cam_module.py` (Grad‑CAM), `dataio.py`/`cryosparc_simulator.py` (synthetic projections), `swd.py` (sliced‑Wasserstein metric). | Mostly peripheral / diagnostic. |

---

## 4. How a training run works

Runs are launched from a dated shell script (`run_2026_0706.sh`) that `source`s `paths`
(dataset name → `.star` path map) and calls `torchrun --nproc_per_node 8 cryofifa.py …`
with a long flag string. A representative invocation (β2‑adrenergic, bin1, half‑map):

```bash
torchrun --nproc_per_node 8 cryofifa.py \
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

The loop (see `cryofifa.py`): for each epoch it selects the current resolution shell,
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
