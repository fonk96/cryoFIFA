#!/bin/bash
ARM=$1; MIX=$2   # MIX = 0103 or 0105
SEED=${SEED:-0}; SUFFIX=${SUFFIX:-}   # env overrides: SEED for a fresh basin, SUFFIX for a new (non-clobbering) dir
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
Date=2026_Jul_06
export OMP_NUM_THREADS=1; export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
save_dir=results/${Date}/mix${MIX}_${ARM}${SUFFIX}/; mkdir -p ${save_dir}
TOPK=""
# 2x2 factorial (marching x loss-rank), matching the uniform curriculum ablation:
#   lrON=both(march+LR)  lrOFF=marchonly  lossrankonly=LR only(no march)  neither=both off
# marching: ON=start_res 80 -> end 8 ; OFF=start_res 8 (full band from ep0)
LR_RAMP="--hard_topk --topk_start_epoch 0 --topk_end_epoch 40 --topk_start_ratio 0.5 --topk_end_ratio 1.0"
SR=80
case $ARM in lossrankonly|neither) SR=8 ;; esac
[ "$ARM" = "lrON" ]         && TOPK="$LR_RAMP"
[ "$ARM" = "lossrankonly" ] && TOPK="$LR_RAMP"                                   # loss-rank, marching OFF
[ "$ARM" = "lrHold" ]       && TOPK="--hard_topk --topk_start_epoch 0 --topk_end_epoch 70 --topk_start_ratio 0.5 --topk_end_ratio 0.5"
[ "$ARM" = "lrHold70" ]     && TOPK="--hard_topk --topk_start_epoch 0 --topk_end_epoch 70 --topk_start_ratio 0.5 --topk_end_ratio 0.7"
# lrOFF (marchonly) and neither: TOPK stays ""
torchrun --master_port 12182 --nproc_per_node 8 --nnode 1 --node_rank=0 ../src/cryofifa_ddpfix.py \
  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 70 --minibatch-size 32 \
  --mask --input_radius 0.4 --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN \
  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1 \
  --particles_meta /sddn/kongfang_work/cryoFIFA/data1/ribo_mix${MIX}.star --using_shift_coords \
  --data_path /sddn/kongfang_work/YANG/Projects/CS-yang/ --decoder_add_layers 2 \
  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 32 --mask_epoch 2 \
  --fourier --imgsize 128 --lowpass --preprocessing_output_channels 2 --skip_read_pose \
  --lmdb_path /data/lmdb/ribo_mix${MIX} --float16 --reduce_input --start_resolution ${SR} --sym \
  --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8 --noise_radius 0.4 \
  --batch_size 32 --preprocess --end_resolution 8.0 --free_gpu_memory 4 --add_distrubance \
  --distrubance_ratio 0.4 --distrubance_epoch 40 --norm6d --uniform --uniform_epoch 45 \
  --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --channels_last --fast_ddp \
  --grad_checkpoint --volume_prior exponential --volume_prior_weight 0.05 --volume_prior_beta 6 \
  --volume_prior_start_radius 0.4 --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore \
  --seed ${SEED} --save_pose --save_checkpoint --checkpoint_interval 10 --checkpoint_keep_all ${TOPK}
