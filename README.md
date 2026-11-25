# cryoFIFA — Project Documentation

## 1. What this project does

cryoFIFA reconstructs a **3D electron‑density map** (and, optionally, a *continuum* or
*discrete set* of conformational states) directly from a stack of noisy 2D cryo‑EM
particle images — **without requiring pre‑determined particle orientations** (ab‑initio).
It is a research reimplementation/extension in the same family as **cryoDRGN**,
**cryoAI**, and **cryoSPARC's ab‑initio**, with two distinctive commitments:


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

### BEFORE TRAINING:
We strongly recommond you to cache particles to LMDB database first.
usage:
```
python  src/lmdbwriter.py --particles_meta ${your_particles.star}  --data_path ${data_path} --db_path ${lmdb_path}${datasetname}
```
## 2. How a training run works

Runs are launched from a shell script under `experiments/` that `source`s `paths`
(dataset name → `.star` path map) and calls `torchrun --nproc_per_node 8 ../src/cryofifa.py …`
with a long flag string. A representative invocation (β2‑adrenergic, bin1, half‑map):

```bash
torchrun --nproc_per_node 8 ../src/cryofifa.py \
  --particles_meta ${B2_act_half2} --lmdb_path /data/lmdb/B2_act_half2 \
  --imgsize 228 --data_path /data/cryoFIFA_data/ \
  --encoder_backbone resnet --encoder_basename resnet18 \
  --decoder_layers 2 --decoder_hidden_dim 256 --activation SIREN --w0 48 --w 32 \
  --fourier --rotation6d --norm6d --sym --sym_factor 8 \
  --start_resolution 40 --end_resolution 4 --mask_steps 64 --mask_epoch 3 \
  --num-epochs 100 --minibatch-size 32 --batch_size 32 --float16 --preprocess \
  --add_distrubance --distrubance_ratio 0.5 --distrubance_epoch 70 --uniform \
  --world_size 8 --free_gpu_memory 8 --render_path results/<run>/
```

