# cryoFIFA — Detailed Code Review

Scope: `/sddn/kongfang_work/cryoFIFA/scripts/`. This review is based on a full read of the
training loop, model library, data‑IO and Fourier stacks, and the `utils`/`untiled` toolbox.
References are `file:line` against the current `scripts/` tree.

## 0. Overall assessment

**What is genuinely good.** The core science is sound and non‑trivial: a correct
projection–slice Fourier forward model, a thoughtful envelope×modulant implicit decoder,
a real coarse‑to‑fine frequency‑marching schedule, a multi‑hypothesis symmetric pose loss,
pose‑uniformity regularisation, relativistic CTF modelling, and serious systems engineering
(LMDB pre‑baking, DDP + AMP `float16`, gradient accumulation, and **dynamic minibatch
resizing driven by live GPU free‑memory** — a pragmatic OOM guard rarely seen in research code).

**What holds it back.** This is a long‑lived single‑author research codebase and it shows:

- **A ~250‑flag argument parser** (`utils.py:get_parser`, ~330 lines) that encodes a vast,
  undocumented configuration space. Many flags gate dead or broken branches. This is the single
  largest maintainability liability.
- **Pervasive copy‑paste.** The same concept is implemented 2–5×: FFT wrappers, CTF, circular
  masks, 6D→rotation orthonormalisation, `which_half_space`, `Symmetrizer`, `image_duplicator`,
  `slicer`, the symmetric losses, and the numpy/torch signal‑processing pair.
- **Snapshot versioning** (`V2402…V2512`, `Pure`) instead of git — no history, no diffs, no branches.
- **Silenced warnings** (`warnings.filterwarnings('ignore')`) hiding real deprecations and shape bugs.
- **Bilingual (English/Chinese) comments** and debug cruft left in (`"What the fuck else here?"`,
  stray hex strings, left‑in `print`s).
- **No unit tests** for the physics (the `test_*.py` files are ad‑hoc smoke scripts), and **no model
  checkpointing or LR schedule** in the production trainer.

The important nuance: **most confirmed bugs live in dead or seldom‑used branches** (downsample
training, `mode='2d'/'both'`, the synthetic‑data simulator, the unused ParNet). The **hot path**
(`cryofifa.py` `mode='3d'` with LMDB + 6D rotation) is largely correct. The risk is that the dead
branches are landmines for the next person who flips the corresponding flag.

---

## 1. Training loop — `cryofifa.py` / `kryoAE_trunk.py` (current)

`cryofifa.py` (688 lines) is the active trainer referenced by `run_2026_0706.sh`;
`kryoAE_trunk.py` (689 lines) is the near‑identical sibling. One monolithic `main()`.

**Strengths**
- Clean amortised‑inference structure; correct Fourier‑slice forward model
  (`rotated_plane_grid = bmm(center_plane_grid, R^T)` → decoder → `view_as_complex` →
  `phase_shift_ft` → ×CTF → ×mask → complex L2). See `cryofifa.py:372‑459`.
- **Dynamic minibatch sizing** (`cryofifa.py:86‑95, 257‑264`): pre‑creates DataLoaders with
  halving batch sizes and steps down when `torch.cuda.mem_get_info` reports low free memory.
- `--reduce_input` gathers only the in‑mask Fourier pixels through the decoder — a real FLOP saving
  during coarse epochs.
- Gradient accumulation to a target `--batch_size` with AMP `GradScaler`.

**Issues**
- **No checkpointing.** There is no `torch.save`/`load_state_dict` anywhere in the trainer — only
  reconstructed MRCs are persisted. A crash loses all weights; you cannot resume, fine‑tune, or run
  inference on new particles without full retraining. (Confirmed by grep across
  `cryofifa*.py`/`kryoAE_trunk.py`.)
- **No LR scheduler.** Fixed Adam LR for the whole run; no warmup/decay despite the strong
  coarse‑to‑fine curriculum that would benefit from one.
- **Dead / broken non‑`3d` branches:** `mode=='both'` render path uses undefined
  `Encoder_2D_Predictor`/`Decoder_2D`; `mode=='2d'/'both'` reference undefined `loss_2d`
  (`kryoAE_trunk.py:571, 580‑583, 621, 624`). Only `mode='3d'` actually runs.
- **`radwn` dead computation:** `radwn=int(imgsize//2*data_apix/args.mask_lowpass)` is immediately
  overwritten by `radwn=16` (`cryofifa.py:109‑111`).
- The pose‑writing block (`cryofifa.py:461‑491`) builds star lines by giant string concatenation
  with per‑element `.cpu().detach().numpy()` calls inside a Python loop — slow and fragile.
- Whole‑volume render decodes slice‑by‑slice in a Python `for slicei in range(imgsize)` loop
  (`cryofifa.py:639‑640`) — fine for periodic snapshots but a latent perf sink at large boxes.
- Heavy commented‑out code (FairScale OSS/ShardedDDP, denoise loss, soft‑mask generation).

**Variants**
- `cryofifa_2026.py` / `cryofifa_back.py`: older loop shapes (e.g. an active `noise_grad` whitening
  block that is commented out in the current file); kept as snapshots.
- `cryoFIFA_Multiclass.py`: discrete heterogeneity — latent `softmax→argmax→one‑hot`
  (`:388‑390`), renders one MRC per class. Note it uses the **non‑normalised** `rotation_6d_to_matrix`
  path and lacks the newer `norm6d`/`DW` refinements.
- `half3d_new.py` (1041 lines, legacy): more pose parameterisations (Euler/quaternion/AF2/6D/2D),
  VAE and auto‑decoder modes, FairScale sharding. **API drift:** it constructs
  `Fullnet(... latent_in_dim=...)` but the current `merged_models.Fullnet` expects `latent_code_dim=`
  — so it would not run against today's library. Also `main()` swaps `num_layers`/`hidden_dim`
  assignments (`:331‑332`) and a render branch references undefined `data_test` (`:951`).

---

## 2. Model library — `merged_models.py`, `PDO_models.py`, `parnet.py`

**`merged_models.py` (595 lines) — core, well‑factored.**
- `SirenNet.forward` uses a **silent `try/except` broadcast** for FiLM modulation
  (`:73‑77`: `x *= mod` else `x *= mod.unsqueeze(1)`) — masks shape bugs; make the shape explicit.
- `Modulator`: `layer_w0 = 48 if is_first else 32` is **dead** (computed, never used, `:91`); a
  Siren‑based modulator variant is commented out ("Failed. Bad output").
- `Fullnet` decoder (`:176‑206`): the `use_symmetrizer` Hermitian‑symmetrisation call is **commented
  out** in `forward` (`:192‑194`) — the flag is plumbed but inert.
- `Feature_Extractor` (`:337‑430`): rotation/shift/latent heads are all commented out (moved to
  `Encode_Predictor`); leftover.
- `Encode_Predictor` (`:435‑469`): `rotation_dims/shift_dims/latent_dims` lists are **built but never
  used** (`:438‑450`).
- `weight_init` uses **deprecated** `nn.init.xavier_normal` (no underscore, `:133`).

**`PDO_models.py` (231 lines) — steerable p8 group‑CNN.** Correct in concept but:
- **Hardcoded `.to('cuda')`** throughout (`get_coef`, steering matrices) → not CPU/MPS‑portable.
- **Steerable kernels recomputed every forward** (`:87‑89`) — no caching; expensive.
- Magic numbers tie it to a fixed layout: `p=8`, `num_outputs=7`, channel `56 = 7*8`,
  and a hardcoded `image_size/32` downsample assumption.

**`parnet.py` (269 lines) — ParNet, unused.** Not imported anywhere; not a `Feature_Extractor`
backbone. Contains a **likely serious bug**: `Fusion.forward` (`:123‑124`) does
`idx = torch.randperm(a.nelement()); a = a.view(-1)[idx].view(a.size())` — a full random permutation
of *every* element across batch/channel/space (destroys all structure; a grouped channel‑shuffle was
presumably intended). Also a non‑ASCII dict key `'çonv'` (`:32,38`). Dead/experimental.

**`cam_module.py`** — Grad‑CAM diagnostic; hardcodes `model.layer4[-1].conv2` so it only works for
ResNet backbones. Peripheral.

**`backup_build_encoder.py`** — archival: one live `class Build_Encoder(nn.Module):` line + ~83
commented lines; `nn` isn't even imported. Would raise `IndentationError` if imported. Delete or move
to an `archive/`.

---

## 3. Data IO & Fourier — `dataio.py`, `new_dataloader.py`, `new_utils.py`, `fourier*.py`, `sigproc*.py`, `swd.py`

**Production path is `LmdbDataSet`** (`new_dataloader.py`), fed by `data_constructor`
(`new_utils.py:46`) after `untiled.Dataloader.get_full_info` parses the star file and
`load_and_drop_particles` bakes particles into LMDB. This path is correct.

**`fourier_utils.py` — the FFT/CTF the pipeline actually uses.**
- Convention: `ft2 = fftshift∘fftn∘ifftshift` (DC‑centered, unnormalised "backward"); `ft3` the 3D
  analogue. `create_ctf` (`:93`) is relativistic (λ from `12.2639/√(V+0.97845e‑6·V²)`), astigmatic,
  returns `-fftshift(ctf)` to match the centered convention. **Correct.**
- **But there are two live CTF implementations** — `create_ctf` (used) and `compute_ctf` (`:33`,
  unused) — with **inconsistent fftshift/sign conventions**, plus a *third* commented‑out `create_ctf`
  copy (`:131‑162`). A trap for anyone who wires up `compute_ctf`.
- **Unit inconsistency for `dfang`:** `get_ctf` (`:78`) converts degrees→radians, while
  `dataio.simulate_data` feeds `create_ctf` raw radians — the two callers disagree on units.
- `torch.meshgrid(...)` without `indexing=` (`:50,97`) → deprecation + latent x/y‑order ambiguity.

**Duplication across the Fourier stack (the biggest cleanup target):**
- `ft2/ift2/ft3/ift3` defined in **both** `fourier_utils.py` (`:195‑207`) and `utils.py`
  (`:1475‑1496`) — the `utils` copies add a `norm=` arg and a `float16→float32` upcast. Which wins
  depends on import order (`from utils import *` then `from fourier_utils import *`), a silent
  divergence hazard.
- `which_half_space` defined **4×** (`fourier_utils`, `new_utils`, `untiled`, `utils`).
- `Symmetrizer` + `where_DC` duplicated verbatim between `utils.py` and `untiled.py`.
- `slicer`/`torch_map_coordinates` duplicated in `dataio.py` and `new_utils.py` (differ only in
  default `mode`).
- `sigproc.py` ≡ `torch_sigproc.py` (numpy vs torch copies of the same 5 functions; a left‑in
  `print(c.size())` in the torch one).
- `dt` duplicated (`fourier_utils`, `utils`).

**A second, incompatible FFT stack:** `fourier.py` ("C. L. Wang 2022") is a pyFFTW
*orthonormal/centered* implementation using the checkerboard `(-1)^(r+c)` trick and Hermitian‑half
`irfftn`, reachable **only** from mask utilities (`new_utils.generate_tight_mask`,
`untiled.lowpass_filter_for_cryoem`). It is a genuinely different convention from the torch `ft2/ft3`
— keeping both invites subtle sign/normalisation mismatches.

**Confirmed data‑path bugs (all in the downsample / simulator branches):**
- `dataio.py:187` — `phase_shift_ft(slices, shifts)` called with 2 args; the imported function needs 4
  → `TypeError` whenever `--shift` is used in `simulate_data`.
- `new_dataloader.py:99` — `crop_input(mrcdata)` called with 1 arg; signature needs 2.
- `new_dataloader.py:114` — uses `self.downsample_imgsize`, never set in `__init__` (only
  `self.downsample_size`) → `AttributeError` when `downsample=True`.
- `new_utils.py:57` — `data_constructor` never forwards `downsample*`/`pose_type` to `LmdbDataSet`, so
  the whole downsample‑training machinery is **silently inert**.
- `new_utils.py:156‑157` — `symmetrize_output` assigns to `.real`/`.imag` of a *slice* of a complex
  tensor (non‑writable views) → **silent no‑op**; the intended Hermitian symmetrisation never happens.
- `fourier.py:46,47,59,60` — `ims.shape[-2:0]` is an empty slice → the shape assertions are vacuously
  true and check nothing.
- `dataio.py:99` — `volue=...` typo (result discarded); and applying a real‑space spherical mask to a
  Fourier volume is conceptually wrong anyway.
- `dataio.py:182` — `if input_pose==None:` on a possible ndarray (should be `is None`).

**`swd.py`** — a correct sliced‑Wasserstein image metric, but **imported into `utils.py` and never
called**, and hardwired to 3‑channel RGB (`range(3)`), so unusable on single‑channel particles as‑is.
Dead import.

**`fts.py` / `ft2.py`** — essentially dead: `fts.syn` is `pass`, `rotate_3d ≡ rotate__patch`,
`fts.rotate` references `self` at module scope; `ft2.py` is an 11‑line argparse stub with no logic.

---

## 4. Toolbox — `utils.py` (1559 lines) & `untiled.py` (973 lines)

**`utils.py`** is the grab‑bag core: the 250‑flag parser, pose/rotation math, symmetry, masks,
low/high‑pass, radial spectral stats, CTF, FFT, and the loss zoo.

- **Rotation math is redundant:** `theta_to_matrix`, `sixd_to_two_orthonormal` (defined **twice**,
  `:550` and `:606`, one a commented duplicate), `gram_schmidt_orthogonalization` — three+ overlapping
  6D‑orthonormalisation routines; plus quaternion (`af2_quater_matrix`, `generate_quater`) and Euler
  paths.
- **Loss zoo, mostly copy‑paste:** five near‑identical symmetric best‑of‑N losses
  (`symmetric_loss`, `symmetric_complex_loss`, `symmetric_complex_L1_loss`,
  `symmetric_complex_wloss`, `complex_W_loss`) and several complex L1/L2 variants.
  **`Complex_log_L2` (`:922`) is byte‑identical to `Complex_L2`** — the "log" is a lie.
  `Complex_L1_Loss` (`:909`) takes `log` of possibly‑negative differences → NaN.
- **No packaged FSC** despite it being central to cryo‑EM QA — only the radial primitives
  (`compute_rotational_average`, `compute_rotational_std`, `create_image_from_radial_profile`) exist,
  and they even disagree on the center convention (`(N-1)//2` vs `N//2`). Half‑map FSC is done
  externally.
- **Confirmed bugs (dead paths):** `expmap` calls undefined `map_to_lie_algebra` (`:446`, `NameError`);
  `data_prior_loss` uses `.float` without `()` (`:491`, `AttributeError`); `gen_shiftmasks_torch`
  leaves `masks` unbound in the `'3d'` branch (`:1196`, `NameError`); `one_hot_encoding` defined twice
  with different signatures (`:14`, `:392`).

**`untiled.py`** is **misnamed** — it does no untiling; it is the star/MRC IO + LMDB baking layer
(imported as `DL`). The hand‑rolled star parser (`get_full_info`) uses whitespace `split(' ')` with
manual `''` removal, `try/except` for control flow, and silent `except: pass` — fragile against
formatting variations. Contains `RX` (broken `np.asmatrix` call), `image_duplicator` using non‑existent
`np.cat` (`:43`), undefined `test_stack_name` (`:330`), `lowpass_filter_for_cryoem` using `fourier.*`
while `import fourier` is commented out (`NameError`), and debug artifacts (`"What the fuck else
here?"`, stray `#9C7135`).

**Helper scripts.** `experiment_report_generator.py` (parse run scripts → PDF of tri‑view MRC slices;
hardcodes `./old_scripts`, `./reports`; a dead `os.remove('reports/.temp*png')` that won't glob).
`kcs2star.py` (cryoSPARC→star; hardcoded `π=3.1415926`; **tab/space mixed indentation** at `:52`).
`cryosparc_simulator.py` is a **copy‑pasted fragment** (starts mid‑function, dozens of undefined
names) — reference snippet only. `filter.py` — `lowpass` computes `ft_radius` but never uses it, so
its cutoff argument is ignored.

---

## 5. Defect summary (quick index)

| Severity | Where | Issue |
|----------|-------|-------|
| High (silent) | `new_utils.py:156`, `new_dataloader.py:99/114`, `new_utils.py:57` | downsample/symmetrize paths silently no‑op or `AttributeError`. |
| High (data) | `fourier_utils.py` `create_ctf` vs `compute_ctf`; `dfang` units | two CTF conventions + a deg/rad mismatch between callers. |
| High (ops) | `cryofifa.py` | no model checkpoint save/load; no LR schedule. |
| Medium | `merged_models.py:73`, `Fullnet` symmetrizer commented out | silent FiLM broadcast; inert symmetry flag. |
| Medium | `utils.py` FFT/CTF/mask duplication; `fourier.py` second FFT stack | convention‑divergence hazard. |
| Low (dead) | `parnet.py:123`, `mode='2d'/'both'`, `expmap`, `swd` import, `fts.py`, `ft2.py`, `backup_build_encoder.py`, `cryosparc_simulator.py` | broken/unused; delete or quarantine. |
| Low (hygiene) | ~250 flags, snapshot versioning, silenced warnings, bilingual/debug comments | maintainability. |

See **OPTIMIZATION.md** for how to prioritise fixing these versus method/perf work.
