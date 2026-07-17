#!/bin/bash
# LOCAL-ON-GLOBAL (2026-07-14): resume the GLOBAL-ONLY (norefine) half1 checkpoint (ep90, ~3.8A) and add
# the local-refine module on top, pushing to higher resolution over more epochs. mask_steps 46 keeps the
# per-step frequency increment identical to the norefine schedule, so the frontier RESUMES at ~3.8A
# (step 30 @ ep90) and marches on to end_resolution 2.5A -- no jump into untrained high-freq shells.
# The resume loads efe/e3dp/dec (global); the fresh local encoders (efe_local/e3dp_local) warm-restart
# their LR at the transition. Data = B2_act_half1 (same as the global run).
Date=2026_Jul_06
source ../configs/paths
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ckpt=results/${Date}/B2_ablation_norefine/checkpoint.pt
save_dir=results/${Date}/B2_localonglobal_end25/
mkdir -p ${save_dir}
torchrun --master_port 12174 --nproc_per_node 8 --nnode 1 --node_rank=0 ../src/cryofifa_ddpfix.py \
  --resume ${ckpt} \
  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 160 --minibatch-size 32 \
  --mask --input_radius 0.4 --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN \
  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 20 --split_rate 1 \
  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ \
  --decoder_add_layers 2 --encoder_activation SIREN --last_activation SIREN --lowpassmask \
  --mask_steps 46 --mask_epoch 3 --fourier --imgsize 228 --lowpass --preprocessing_output_channels 2 \
  --skip_read_pose --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 40 \
  --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8 --noise_radius 0.4 \
  --batch_size 32 --preprocess --end_resolution 2.5 --free_gpu_memory 8 --add_distrubance \
  --distrubance_ratio 0.4 --distrubance_epoch 90 --norm6d --uniform --uniform_epoch 160 \
  --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last \
  --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore \
  --seed 0 --save_checkpoint --checkpoint_interval 20 \
  --downsample_train --downsample_radius 48 --local_refine_scale 0.3 --downsample_soft_crop 0.15 --fade_in_epochs 1
