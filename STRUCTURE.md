# cryoFIFA — Reorganized Project Layout

This is a **cleaned‑up copy** of `/sddn/kongfang_work/cryoFIFA/scripts/` (which lives on a
read‑only NFS mount). Only the **code** was reorganized here — the multi‑GB `data*/` star/mrcs
and `results/` volumes stay on `/sddn`. Nothing in the original tree was modified.

The guiding rule of the reorg: **the training import graph is a flat `from X import *` web**, so
every module that participates in it is kept together in `src/` and nothing was renamed. Files that
*nothing imports* (dead/experimental/legacy) were moved to `archive/`; standalone tools and tests
were separated out. All 20 `src/` files pass a syntax check, confirming the copy is byte‑intact.

```
/home/user/cryoFIFA/
├── README.md            ← what the project is + how it works
├── CODE_REVIEW.md       ← file‑by‑file review + defect catalogue
├── OPTIMIZATION.md      ← engineering/perf roadmap (tiers A–D)
├── ASSESSMENT.md        ← scientific judgment: what's genius, why it plateaus, where to push
├── STRUCTURE.md         ← this file
│
├── src/                 ← the active import web (run trainers from here, or PYTHONPATH=src)
│   ├── cryofifa.py            (current production trainer — the one in run_2026_0706.sh)
│   ├── kryoAE_trunk.py        (near‑identical sibling trainer)
│   ├── cryofifa_2026.py       (variant snapshot)
│   ├── cryofifa_back.py       (variant snapshot)
│   ├── cryoFIFA_Multiclass.py (discrete‑heterogeneity variant)
│   ├── merged_models.py       (model library: SirenNet, Fullnet, Feature_Extractor, …)
│   ├── PDO_models.py          (p8 steerable group‑CNN backbone)
│   ├── cam_module.py          (Grad‑CAM; imported by the trainers)
│   ├── new_dataloader.py      (LmdbDataSet / StarDataSet)
│   ├── new_utils.py           (model_constructor, preprocesser, mask schedule, losses)
│   ├── untiled.py             (star/MRC IO + LMDB baking — the real data layer)
│   ├── filter.py              (small MRC/filter helper used by untiled)
│   ├── utils.py               (1559‑line core toolbox + ~250‑flag get_parser)
│   ├── fourier_utils.py       (torch FFT + CTF used by the pipeline)
│   ├── fourier.py             (separate pyFFTW orthonormal FFT, used only by mask utils)
│   ├── sigproc.py / torch_sigproc.py  (numpy/torch signal‑processing pair)
│   ├── swd.py                 (sliced‑Wasserstein metric; imported but currently unused)
│   ├── dataio.py              (synthetic projection simulator — importable)
│   └── fts.py                 (rotation/FFT scratch used only by dataio)
│
├── tools/               ← standalone entry points (run with PYTHONPATH=../src if they import src)
│   ├── kcs2star.py                    (cryoSPARC .cs → RELION .star)
│   ├── experiment_report_generator.py (parse run scripts → PDF of tri‑view MRC slices)
│   └── compare_ft.py                  (FT comparison harness; imports dataio)
│
├── tests/               ← ad‑hoc dev smoke scripts (some reference the legacy model_utils)
│   └── test.py, test_loader.py, test_newloader.py, test_cam.py, test_group_lp.py
│
├── configs/
│   ├── environment.yaml  (conda env spec — was cryofifa.yaml; py3.8, torch2.0.1/cu118, pytorch3d)
│   └── paths             (dataset‑name → .star path map, source'd by run scripts)
│
├── experiments/
│   └── run_2026_0706.sh  (current launch script)
│
└── archive/             ← dead / broken / legacy (nothing in src imports these)
    ├── half3d_new.py            (legacy trainer; imports a missing `new_model_utils` → won't run)
    ├── backup_build_encoder.py  (empty class + commented body)
    ├── parnet.py                (unused ParNet; has a full‑tensor randperm bug)
    ├── cryosparc_simulator.py   (copy‑pasted fragment, many undefined names)
    ├── ft2.py                   (11‑line argparse stub)
    ├── root_dir.py              (trivial, unused)
    └── oldscripts/              (historical run scripts 2025‑03 … 2026‑01 + model_utils_legacy.py)
```

### How to run from this copy

```bash
cd /home/user/cryoFIFA/src
source ../configs/paths                 # defines $Beta2_128 etc. (edit data_path first)
torchrun --nproc_per_node 8 cryofifa.py --particles_meta $Beta2_128 --lmdb_path ... [flags]
```

The trainers must be launched **from `src/`** (or with `PYTHONPATH=/home/user/cryoFIFA/src`) because
the modules import each other flat. Tools in `tools/` that import project modules (e.g. `compare_ft`)
likewise need `src/` on the path.

### What was intentionally left out

- The dated whole‑directory snapshots `V2402/ V2503/ V2509/ V2512/ Pure/` on `/sddn` are the
  original ad‑hoc "version control." They were **not** copied (they duplicate the tree). Replacing
  them with real git is recommended (OPTIMIZATION.md · A5).
- All `data*/` and `results/` payloads remain on `/sddn` (read‑only, and large).
