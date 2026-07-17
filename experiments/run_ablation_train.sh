#!/bin/bash
# LOCAL-REFINE ABLATION (2026-07-14): B2_bin1_half1, box 228, mask_steps 32 x mask_epoch 3,
# end_resolution 3.5A, 90 epochs. ARM = norefine | localrefine.
#   localrefine = --downsample_train two-stage module: coarse global encoder on the (soft) Fourier-cropped
#     image, then a SECOND "local" encoder refines poses at full resolution once the frontier radius passes
#     downsample_radius (48px -> engages ~ep63, ~27 epochs of local refine). Everything else identical, so
#     the difference is attributable to the local-refine module alone.
ARM=$1
Date=2026_Jul_06
# env overrides (default = original values -> existing arms unchanged). Used by the end_res-5 experiment.
NEPOCH=${NEPOCH:-90}; MSTEPS=${MSTEPS:-32}; ENDRES=${ENDRES:-3.5}; DISTEP=${DISTEP:-60}; UNIEP=${UNIEP:-90}; SUFFIX=${SUFFIX:-}
HALF=${HALF:-half1}   # half1 (default, unchanged) | half2 -> B2_act_half2 star + /data/lmdb/B2_act_half2
KEEPALL=${KEEPALL:-}  # KEEPALL=1 -> keep a checkpoint at EVERY interval (checkpoint_ep{15,30,...}.pt) for loss-trajectory tracking
RESUME=${RESUME:-}    # RESUME=<checkpoint.pt> -> continue training from it (restores epoch -> marching continues)
if [ ! -d results/${Date} ] ; then mkdir -p results/${Date}; chmod -R 777 results/${Date}; fi
source ../configs/paths
STAR_VAR="B2_act_${HALF}"; STAR="${!STAR_VAR}"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
save_dir=results/${Date}/B2_ablation_${ARM}${SUFFIX}/
mkdir -p ${save_dir}
EXTRA=""
if [ "$ARM" = "localrefine" ]; then
  EXTRA="--downsample_train --downsample_radius 48 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15"
fi
if [ "$ARM" = "coarseonly" ]; then
  # coarse encoder (96px) predicts pose for ALL epochs; local full-res encoder NEVER runs. marching still drives decoder.
  EXTRA="--downsample_train --downsample_radius 48 --coarse_only --downsample_soft_crop 0.15"
fi
if [ "$ARM" = "coarseonly32" ]; then
  # smaller coarse crop (64px, radius 32) -> cheaper encoder, lower-res pose input. tests how small we can go.
  EXTRA="--downsample_train --downsample_radius 32 --coarse_only --downsample_soft_crop 0.15"
fi
torchrun --master_port 12170 --nproc_per_node 8 --nnode 1 --node_rank=0 ../src/cryofifa_ddpfix.py \
  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs ${NEPOCH} --minibatch-size 32 \
  --mask --input_radius 0.4 --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN \
  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 15 --split_rate 1 \
  --particles_meta ${STAR} --using_shift_coords --data_path /data/cryoFIFA_data/ \
  --decoder_add_layers 2 --encoder_activation SIREN --last_activation SIREN --lowpassmask \
  --mask_steps ${MSTEPS} --mask_epoch 3 --fourier --imgsize 228 --lowpass --preprocessing_output_channels 2 \
  --skip_read_pose --lmdb_path /data/lmdb/B2_act_${HALF} --float16 --reduce_input --start_resolution 40 \
  --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8 --noise_radius 0.4 \
  --batch_size 32 --preprocess --end_resolution ${ENDRES} --free_gpu_memory 8 --add_distrubance \
  --distrubance_ratio 0.4 --distrubance_epoch ${DISTEP} --norm6d --uniform --uniform_epoch ${UNIEP} \
  --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last \
  --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore \
  --seed 0 --save_checkpoint --checkpoint_interval 15 $([ -n "$KEEPALL" ] && echo --checkpoint_keep_all) $([ -n "$RESUME" ] && echo --resume $RESUME) ${EXTRA}
