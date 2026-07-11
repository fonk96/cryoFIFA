
Date=2026_Jul_06
if [ ! -d results/${Date} ] ; then
mkdir -p results/${Date}
chmod -R 777 results/${Date}
fi 
#export CUDA_VISIBLE_DEVICES=4,5,6,7

source ../configs/paths
#save_dir=results/${Date}/save_path/


export OMP_NUM_THREADS=1
# reduce CUDA memory fragmentation / OOM (free, no code change). NOTE: expandable_segments needs
# torch>=2.1; this env has torch 2.0.1, so use max_split_size_mb (supported since 1.10) instead.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# export NCCL_IB_DISABLE=0
# #export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=ib0

# good reproduce, merged full
# save_dir=results/${Date}/B128_full/
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 150 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.37 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${Beta2_128} --using_shift_coords --data_path ${data_path} --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 3 --fourier  --imgsize 128  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/Beta2_128 --float16 --reduce_input --start_resolution 40 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 4  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.5 --distrubance_epoch 70 --norm6d

# save_dir=results/${Date}/B128_half1_run1/
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 150 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.37 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${Beta2_128_half1} --using_shift_coords --data_path ${data_path} --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 3 --fourier  --imgsize 128  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/Beta2_128 --float16 --reduce_input --start_resolution 40 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 4  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.5 --distrubance_epoch 70 --norm6d --uniform


# /data/cryoFIFA_data/
# B2_act_half1
# save_dir=results/${Date}/B2_bin1_half1_run1/
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 150 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 3 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.5  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 90 --norm6d --uniform --downsample_train --downsample_radius 32 --local_refine_scale 0.3 --fade_in_epochs 1

# NOTE: half2 block below was commented out to isolate the new upgrade test.
#       It also had a typo ("--reconstruct_radius 0.38--w0" -> missing space) that would crash argparse; fixed here for when you re-enable it.
# save_dir=results/${Date}/B2_bin1_half2_run1/
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 150 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half2} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 3 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half2 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.5  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 90 --norm6d --uniform --downsample_train --downsample_radius 32 --local_refine_scale 0.3 --fade_in_epochs 1


# =====================================================================================
# UPGRADE TEST (2026-07-06): B2 half1 with the 5 new upgrades, into a NEW output dir so it
# can be compared side-by-side against the baseline results/${Date}/B2_bin1_half1_run1/.
#   #1 checkpointing : --save_checkpoint  (writes ${save_dir}/checkpoint.pt every save-interval)
#                      resume a crashed run by adding:  --resume ${save_dir}/checkpoint.pt
#   #2 soft masks    : --soft_mask --soft_mask_width 0.03   (raised-cosine edges -> kill ringing)
#   #3 volume prior  : --volume_prior gaussian --volume_prior_weight 0.01 --volume_prior_power 2
#                      (Gaussian/Wiener shell prior; if HF noise persists raise the weight ~3-10x,
#                       if the map looks over-smoothed lower it. Watch the printed Error_3d.)
#   #4 non-negativity: --enforce_non_neg   (clamp rendered density >=0)
#   #5 normalization : --norm_zscore       (zero-mean/unit-std inside mask instead of min-max)
# Everything else is identical to the half1 baseline, so differences are attributable to the flags.
# save_dir=results/${Date}/B2_bin1_half1_volprior/   # v1 (completed) -- commented so run.sh runs v2 only
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 150 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 3 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.5  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 90 --norm6d --uniform --downsample_train --downsample_radius 32 --local_refine_scale 0.3 --fade_in_epochs 1  --soft_mask --soft_mask_width 0.03 --volume_prior gaussian --volume_prior_weight 0.01 --volume_prior_power 2 --enforce_non_neg --norm_zscore

# =====================================================================================
# UPGRADE v2 (2026-07-07): fixes for late-training degradation + TM-region worsening.
#  POSE REGULARIZATION KEPT ON ALL RUN (was the root cause of late collapse):
#    --uniform_epoch 150            (was default 100 -> pose-uniformity now on the whole run)
#    --distrubance_epoch 150 --distrubance_end_ratio 0.05  (extend disturbance, small residual late)
#  AMORTIZED-FRIENDLY POSE (no SO(3) search; keeps single-pass inference):
#    --soft_min --soft_min_temp 0.1 (marginalize the in-plane sym hypotheses instead of hard best-of-N)
#  DOWNSAMPLE / GLOBAL INFO (fade-in now actually freezes global while local warms up):
#    --downsample_radius 48         (was 32: global encoder sees 96px not 64px -> more global info,
#                                    and local refinement starts later so global converges first)
#    --fade_in_epochs 5             (real warm-up window; gate re-keyed to the radius transition)
#  cryoSPARC-STYLE VOLUME PRIOR (stronger, concentrated on the noise band):
#    --volume_prior exponential --volume_prior_weight 0.05 --volume_prior_beta 6 --volume_prior_start_radius 0.4
#    (exp(beta*k) envelope, 0 below 0.4*Nyquist so it leaves the resolved band alone and only damps noise)
#  Imaging fixes retained: --soft_mask --enforce_non_neg --norm_zscore.
#  NOTE: end_resolution 2.5 is aggressive for a ~50kDa GPCR; if HF noise persists, drop to ~3.5.
# save_dir=results/${Date}/B2_bin1_half1_v2/   # v2 REVERTED -- --soft_min blended the in-plane sym
# hypotheses and stalled contrast (core stuck ~1.2 vs v1's 2.0 by ep90). Do NOT use --soft_min with the
# in-plane best-of-N sym loss; the hard min is load-bearing for ab-initio convergence.
# torchrun ... (v2 line archived; see git/history) ... --soft_min ... --volume_prior exponential ...

# =====================================================================================
# UPGRADE v3 (2026-07-07): disciplined recovery. = the known-good v1 config, changing ONLY the
# pose-regularization SCHEDULE (the real fix for late-training degradation). No soft_min; downsample
# radius, fade-in, and volume prior all reverted to v1 values so this run is attributable to one change.
#   --uniform_epoch 150            (was default 100 -> keep pose-uniformity on the whole run)
#   --distrubance_epoch 150 --distrubance_end_ratio 0.05   (extend disturbance with small residual)
# Everything else identical to v1 (soft_mask, gaussian prior@0.01, non_neg, z-score, downsample_radius 32).
# save_dir=results/${Date}/B2_bin1_half1_v3/   # v3 SUPERSEDED by v4 -- v3 lagged v1 from ep30 (Fourier-crop
# double-encoder + an over-strong mean-normalized prior). Commented so run.sh runs v4 only.
# torchrun ... --downsample_train --downsample_radius 32 ... --volume_prior gaussian ...  (archived)

# =====================================================================================
# UPGRADE v4 (2026-07-07): STRIP BACK TO THE CLEAN CORE METHOD.
#  - NO --downsample_train: the double-encoder fed its global branch a Fourier-CROP of the image
#    (downsample_input hard-crops the central FT box -> sinc ringing in the cropped input). Removing it
#    uses the single, full-resolution encoder path (no crop, no double encoder).
#  - NO volume prior: removes the confound (the mean-normalized weight-map made it ~2x stronger than the
#    original v1 and it was suppressing contrast without a clear win).
#  - Keep pose uniformity on the whole run (--uniform_epoch 150) to stop the late orientation collapse,
#    with the v1 disturbance schedule (--distrubance_epoch 90) so mid-training contrast isn't slowed.
#  - Keep the imaging fixes: --soft_mask --enforce_non_neg --norm_zscore.
# This is the clean A/B answer to "does the Fourier-crop double-encoder hurt?": compare vs v1/v3.
# save_dir=results/${Date}/B2_bin1_half1_v4/   # v4 retired: better early contrast than v1 but still lost resolution -> moving to v5 (soft-crop downsample).
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 150 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 3 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.5  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 90 --norm6d --uniform --uniform_epoch 150 --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore
# checkpointing (#1) implemented & verified; left OFF for this test. To enable + auto-resume, append:
#   --save_checkpoint --resume ${save_dir}/checkpoint.pt

# =====================================================================================
# UPGRADE v5 (2026-07-07): SOFT Fourier-crop test -- can we keep downsample_train's speed WITHOUT the
# ringing? = v4's clean config + downsample_train back on, but --downsample_soft_crop 0.15 apodizes the
# Fourier crop (raised-cosine taper over the outer 15%) instead of the hard box. Verified on a disk to
# cut solvent ringing ~31%. Compare v5 vs v4 (does soft-crop downsample match no-downsample?) and vs
# v1 (does soft-crop rescue the hard-crop downsample path?).
# v5 (box-228 soft-crop) RETIRED: box-228 -> end_resolution 2.5 marches into noise for a ~50kDa GPCR ->
# fragmented map. The proven path is box-128 (band-limited to ~4A). See B2_128 block below.
# save_dir=results/${Date}/B2_bin1_half1_v5/
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 100 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 3 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.5  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 90 --norm6d --uniform --uniform_epoch 150 --downsample_train --downsample_radius 32 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore

# =====================================================================================
# BOX-128 (2026-07-07): the user's PROVEN-GOOD recipe (cryoSPARC crop-to-128 particles, band-limited
# to ~4A). Exactly the archived Beta2_128_half1 config + the safe output-only fixes (soft_mask/non_neg/
# z-score, which change only the rendered MRC, not convergence). --num-epochs 100 diagnostic (reaches
# ~5.5A; run 150 for the full ~4A "very good" map). This should give a coherent envelope, unlike box-228.
# save_dir=results/${Date}/B2_128_half1_diag/   # separate box-128 dataset run -- superseded by the
# internal-crop approach below (box-228 data, crop-to-128 for the global encoder).
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 100 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.37 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${Beta2_128_half1} --using_shift_coords --data_path ${data_path} --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 3 --fourier  --imgsize 128  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/Beta2_128 --float16 --reduce_input --start_resolution 40 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 4  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.5 --distrubance_epoch 70 --norm6d --uniform --uniform_epoch 100 --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore

# =====================================================================================
# "TRAIN ON 128 FIRST, THEN LOCAL-REFINE ON 228" via existing downsample_train, done RIGHT (2026-07-07).
# KEY CHANGE vs all previous downsample_train runs: --downsample_radius 64  (crop box = 2*radius = 128,
#   Nyquist ~3.85A = the cryoSPARC box-128 scale). Previous runs used 32/48 -> box 64/96 (~7.7/5.1A) =
#   too coarse for the global encoder to carry pose to high res.
# Plus the fixes made since: soft Fourier crop (--downsample_soft_crop 0.15, no sinc ringing in the
#   global-encoder input), the fade-in freeze re-keyed to the actual radius transition (was a no-op),
#   local-predictor LR consistent (lr*local_refine_scale), --end_resolution 3.5 (not 2.5 -> stop before
#   noise), pose-uniformity kept on all run, and the output fixes (soft_mask/non_neg/z-score).
# DIAGNOSTIC: --num-epochs 100 -> stays in the box-128 COARSE phase (local refine only engages when the
#   frontier radius > 64px, ~ep129 at mask_epoch 3). So this run tests ONLY whether box-128 conditioning
#   (radius 64) gives a coherent map, isolating the radius-64-vs-32/48 question. Run 150 next for the
#   box-228 local-refine phase to push toward 3.5A.
# SCHEDULE COORDINATION (was wrong before): marching completes at mask_steps*mask_epoch = 48*3 = 144
# epochs, so --num-epochs must be >= ~145 to actually REACH end_resolution 3.5 (100 was an early stop
# at ~4.8A). --distrubance_epoch 120 so pose disturbance covers the box-128 coarse ab-initio and decays
# out just before the box-228 local-refine (radius crosses 64px ~ep129) -- you don't want pose jitter
# during high-res refinement. --uniform_epoch 150 keeps pose-uniformity on the whole run.
# save_dir=results/${Date}/B2_ds64_128first/   # superseded by the longer + LR-decay run below
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 150 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 3 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 3.5  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 120 --norm6d --uniform --uniform_epoch 150 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 3 --downsample_soft_crop 0.15 --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore

# =====================================================================================
# LONGER RUN + COSINE LR DECAY (2026-07-07). New output dir B2_ds64_ms64_lrcos (not reused).
#  - mask_steps 64 (finer marching), mask_epoch 3 -> march completes 64*3 = 192 epochs; --num-epochs 200.
#  - --lr_schedule cosine --lr_warmup --min_lr 1e-5 : LR warms 0..1e-4 over 5 ep, then cosine-decays to
#    1e-5 by ep200 (reaches min only at the end, so high-res shells opening ~ep174-192 are still learned;
#    ~1.4e-5 at the box-228 local-refine transition, good for precise pose refinement).
#  - Coordinated: --distrubance_epoch 150 (covers box-128 coarse ab-initio, decays out before the
#    box-228 refine at ~ep174), --uniform_epoch 200. Otherwise identical to the ds64 (crop-to-128) run.
# save_dir=results/${Date}/B2_ds64_ms64_lrcos2/   # superseded by the faster mask_epoch=2 run below
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 200 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 64 --mask_epoch 3 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 3.5  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 150 --norm6d --uniform --uniform_epoch 200 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 3 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore

# =====================================================================================
# FASTER (mask_epoch 2) + render mask3d FIX (2026-07-07). New dir B2_ds64_ms64_me2.
#  - mask_epoch 2 (was 3): march completes at mask_steps*mask_epoch = 64*2 = 128 epochs -> --num-epochs 140
#    (128 + short settle). ~30% fewer epochs than the ms64/me3 run.
#  - Render now low-passes at the TRAINING FRONTIER as a 3D spherical soft mask (cryofifa.py fix) -> no
#    more untrained-extrapolation speckle at the box centre (the rec_3d_120_frontierLP.mrc look, but live).
#  - Coordinated for me2: box-228 local-refine engages ~ep116; --distrubance_epoch 100 (ends before it),
#    --uniform_epoch 140. Cosine LR reaches 1e-5 by ep140. Everything else = the ds64 crop-to-128 run.
# save_dir=results/${Date}/B2_ds64_ms64_me2/   # DEADLOCKED at the box-228 transition (~ep116); superseded.
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 140 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 64 --mask_epoch 2 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 3.5  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 100 --norm6d --uniform --uniform_epoch 140 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 3 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore

# =====================================================================================
# NEXT-VERSION (2026-07-07): lighter schedule + Tier 0 profile + Tier 1 fast-DDP. New dir.
#  - mask_steps 48, mask_epoch 2 -> march completes 48*2 = 96 epochs; --num-epochs 110 (short settle).
#  - box-228 local-refine engages ~ep88 (radius crosses 64px); --distrubance_epoch 80 (ends before it),
#    --uniform_epoch 110, --fade_in_epochs 1 (was 3, per your note).
#  - --profile --profile_iters 20 : Tier-0 op breakdown at ep0 (table -> log, trace -> render_path).
#  - --fast_ddp : Tier-1 no_sync + bucket_view (identical math, less all-reduce).
#  WATCH: first render (ep10) and the ep88 transition -- the previous run deadlocked at the transition.
# save_dir=results/${Date}/B2_ds64_ms48_me2/   # reached ep94 OK; superseded by the ms64/end3.0 + Tier-2 run below.
# torchrun ... (archived) ... --mask_steps 48 ... --end_resolution 3.5 ... --profile --fast_ddp ...

# =====================================================================================
# NEXT VERSION (2026-07-07): Tier-2 perf (channels_last + activation checkpointing) + expandable_segments,
# pushed to end_resolution 3.0, finer/longer marching. HALF1 then HALF2 (for gold-standard FSC).
#   Schedule: mask_steps 64, mask_epoch 2 -> march completes 128 -> --num-epochs 130. box-228 local-refine
#     engages ~ep98; --distrubance_epoch 90 (ends before it), --uniform_epoch 130, --fade_in_epochs 1.
#   Perf (result-preserving except BN-stat drift from checkpoint): --channels_last --grad_checkpoint
#     --fast_ddp, plus PYTORCH_CUDA_ALLOC_CONF=expandable_segments (env, top of file).
#   --profile on HALF1 only (op + memory tables -> log; trace -> render_path).



    # 185 +save_dir=results/${Date}/B2_h1_ms64_ckpt/                          
    #   186 +torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 
    # --render_path ${save_dir} --num-epochs 130 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activ
    #       +ation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half1} --u
    #       +sing_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmas
    #       +k --mask_steps 64 --mask_epoch 2 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /da
    #       +ta/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_siz
    #       +e 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 3.0  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 
    #       +--distrubance_epoch 90 --norm6d --uniform --uniform_epoch 130 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in
    #       +_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --gra
    #       +d_checkpoint --profile --profile_iters 20 --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore                              
                                                                                                                                      
    #   188 +# ---- HALF2 (queued; runs after HALF1 finishes -- gives the second half-map for FSC) ----                                                
    #   189 +save_dir=results/${Date}/B2_h2_ms64_ckpt/                                                                                                 
    #   190 +torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 
    #       +--render_path ${save_dir} --num-epochs 130 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activ
    #       +ation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half2} --u
    #       +sing_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmas
    #       +k --mask_steps 64 --mask_epoch 2 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /da
    #       +ta/lmdb/B2_act_half2 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_siz
    #       +e 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 3.0  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 
    #       +--distrubance_epoch 90 --norm6d --uniform --uniform_epoch 130 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in
    #       +_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --gra
    #       +d_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore    



# push end_resolution to 2.165

# # ---- HALF1 ----
# save_dir=results/${Date}/B2_h1_ms64_End2/
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 130 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 64 --mask_epoch 2 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.165  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 90 --norm6d --uniform --uniform_epoch 130 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --grad_checkpoint  --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore

# ---- HALF2 (queued; runs after HALF1 finishes -- gives the second half-map for FSC) ----
# save_dir=results/${Date}/B2_h2_ms64_End2/
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 130 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half2} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 64 --mask_epoch 2 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half2 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.165  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 90 --norm6d --uniform --uniform_epoch 130 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore


# save_dir=results/${Date}/B2_h2_ms64_End2_start40/
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 130 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half2} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 64 --mask_epoch 2 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half2 --float16 --reduce_input --start_resolution 40 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.165  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 90 --norm6d --uniform --uniform_epoch 130 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore


# save_dir=results/${Date}/B2_h1_ms64_End2_start40/
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 130 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 64 --mask_epoch 2 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 40 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.165  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 90 --norm6d --uniform --uniform_epoch 130 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore


# # ---- HALF1 ----
# save_dir=results/${Date}/B2_h1_ms64_End2/
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 130 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 64 --mask_epoch 2 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.165  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 90 --norm6d --uniform --uniform_epoch 130 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --grad_checkpoint  --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore

# ---- HALF2  ----
# save_dir=results/${Date}/B2_h2_ms64_End2/
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 130 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half2} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 64 --mask_epoch 2 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half2 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.165  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 90 --norm6d --uniform --uniform_epoch 130 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore


# save_dir=results/${Date}/B2_h1_ms64_ckpt_end22/                          
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 30 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 2 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.2  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.7 --distrubance_epoch 90 --norm6d --uniform --uniform_epoch 130 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore                              
                                                                                                                                      

# save_dir=results/${Date}/B2_h2_ms64_ckpt_end22/                          
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 30 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half2} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 2 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half2 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.2  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.7 --distrubance_epoch 90 --norm6d --uniform --uniform_epoch 130 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore      


# save_dir=results/${Date}/B2_h1_ms64_ckpt_end25_full/                          
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 100 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 20 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 2 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 40 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.5  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.7 --distrubance_epoch 90 --norm6d --uniform --uniform_epoch 130 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore                              
                                                                                                                                      

# save_dir=results/${Date}/B2_h2_ms64_ckpt_end25_full/                          
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 100 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 20 --split_rate 1  --particles_meta ${B2_act_half2} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 2 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half2 --float16 --reduce_input --start_resolution 40 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.5  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.7 --distrubance_epoch 90 --norm6d --uniform --uniform_epoch 130 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore      

#save_dir=results/${Date}/B2_h1_ms64_ckpt_end26/                          
#torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 150 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 15 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 3 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 40 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.6  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.6 --distrubance_epoch 100 --norm6d --uniform --uniform_epoch 150 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore                              
                                                                                                                                      

#save_dir=results/${Date}/B2_h2_ms64_ckpt_end26/                          
#torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 150 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 15 --split_rate 1  --particles_meta ${B2_act_half2} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 3 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half2 --float16 --reduce_input --start_resolution 40 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.6  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.6 --distrubance_epoch 100 --norm6d --uniform --uniform_epoch 150 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore    




#save_dir=results/${Date}/B2_h1_end2_ddpfix/
#torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa_ddpfix.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 150 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 15 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 3 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 40 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.165 --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.6 --distrubance_epoch 100 --norm6d --uniform --uniform_epoch 150 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore


#save_dir=results/${Date}/B2_h2_end2_ddpfix/
#torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa_ddpfix.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 150 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 15 --split_rate 1  --particles_meta ${B2_act_half2} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 48 --mask_epoch 3 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half2 --float16 --reduce_input --start_resolution 40 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.165 --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.6 --distrubance_epoch 100 --norm6d --uniform --uniform_epoch 150 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore 
