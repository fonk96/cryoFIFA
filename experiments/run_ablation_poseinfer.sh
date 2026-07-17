#!/bin/bash
# Best-of-8 pose inference for the local-refine ablation. Architecture MUST match run_ablation_train.sh
# (the localrefine checkpoint carries the local encoders). Resumes at ep90 -> frontier radius > 48 ->
# the local-refine correction IS applied to the exported poses for the localrefine arm.
ARM=$1
Date=2026_Jul_06
MSTEPS=${MSTEPS:-32}; ENDRES=${ENDRES:-3.5}; SUFFIX=${SUFFIX:-}   # match run_ablation_train.sh for the end_res-5 variant
HALF=${HALF:-half1}   # half1 (default) | half2
OUTSUF=${OUTSUF:-}; EXPORT_LOSS=${EXPORT_LOSS:-}   # OUTSUF: separate output dir w/o changing the ckpt; EXPORT_LOSS=1: dump per-particle loss
CKPT_NAME=${CKPT_NAME:-checkpoint.pt}   # resume a specific epoch checkpoint (e.g. checkpoint_ep30.pt) for loss-trajectory export
source ../configs/paths
STAR_VAR="B2_act_${HALF}"; STAR="${!STAR_VAR}"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ckpt=results/${Date}/B2_ablation_${ARM}${SUFFIX}/${CKPT_NAME}
save_dir=results/${Date}/B2_ablation_${ARM}${SUFFIX}${OUTSUF}_poses/
mkdir -p ${save_dir}
EXTRA=""
if [ "$ARM" = "localrefine" ]; then
  EXTRA="--downsample_train --downsample_radius 48 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15"
fi
if [ "$ARM" = "coarseonly" ]; then
  EXTRA="--downsample_train --downsample_radius 48 --coarse_only --downsample_soft_crop 0.15"
fi
if [ "$ARM" = "coarseonly32" ]; then
  EXTRA="--downsample_train --downsample_radius 32 --coarse_only --downsample_soft_crop 0.15"
fi
[ -n "$EXPORT_LOSS" ] && EXTRA="$EXTRA --loss_export"   # per-particle loss -> loss_infer_rank*.txt
[ -n "$EXPORT_CONF" ] && EXTRA="$EXTRA --conf_export"   # per-particle best-of-8 confidence -> conf_infer_rank*.txt
torchrun --master_port 12171 --nproc_per_node 8 --nnode 1 --node_rank=0 ../src/cryofifa_ddpfix.py \
  --pose_infer --resume ${ckpt} \
  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 200 --minibatch-size 32 \
  --mask --input_radius 0.4 --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN \
  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1 \
  --particles_meta ${STAR} --using_shift_coords --data_path /data/cryoFIFA_data/ \
  --decoder_add_layers 2 --encoder_activation SIREN --last_activation SIREN --lowpassmask \
  --mask_steps ${MSTEPS} --mask_epoch 3 --fourier --imgsize 228 --lowpass --preprocessing_output_channels 2 \
  --skip_read_pose --lmdb_path /data/lmdb/B2_act_${HALF} --float16 --reduce_input --start_resolution 40 \
  --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8 --noise_radius 0.4 \
  --batch_size 32 --preprocess --end_resolution ${ENDRES} --free_gpu_memory 8 --add_distrubance \
  --distrubance_ratio 0.4 --distrubance_epoch 60 --norm6d --uniform --uniform_epoch 90 \
  --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last \
  --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore ${EXTRA}
