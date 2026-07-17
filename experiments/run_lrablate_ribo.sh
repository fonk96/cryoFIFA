#!/bin/bash
# LOCAL-REFINE ABLATION on ribosome sim (box 128, SNR 0.5). Part A of the reviewer local-refine study.
# 3 arms (everything else identical -> difference attributable to the local-refine module):
#   standard    = single full-res encoder throughout (no downsample_train) -- the reference
#   coarseonly  = downsample_train, refiner OFF (--local_refine_scale 0): pose FROZEN at the coarse band
#                 once frontier radius > downsample_radius -> isolates "no local refinement"
#   coarselocal = downsample_train + refiner ON (--local_refine_scale 0.3) + --log_refine_delta (Part B data)
ARM=$1; MODE=${2:-train}
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
Date=2026_Jul_06
export OMP_NUM_THREADS=1; export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
MODEFLAGS=""
if [ "$MODE" = "infer" ]; then
  save_dir=results/${Date}/lrab_ribo_${ARM}_poses/
  MODEFLAGS="--pose_infer --resume results/${Date}/lrab_ribo_${ARM}/checkpoint_ep70.pt --num-epochs 71"  # resume@70 -> range(70,71)=1 pass (was 70 -> empty range, exported nothing)
else
  save_dir=results/${Date}/lrab_ribo_${ARM}/
fi
mkdir -p ${save_dir}
EXTRA=""
case $ARM in
  coarseonly)  EXTRA="--downsample_train --downsample_radius 32 --local_refine_scale 0 --downsample_soft_crop 0.15" ;;
  coarselocal) EXTRA="--downsample_train --downsample_radius 32 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --log_refine_delta" ;;
  scale*)      SC=${ARM#scale}; EXTRA="--downsample_train --downsample_radius 32 --local_refine_scale ${SC} --fade_in_epochs 1 --downsample_soft_crop 0.15 --log_refine_delta" ;;  # Part C sweep: ARM=scale0.05 etc.
esac
torchrun --master_port 12184 --nproc_per_node 8 --nnode 1 --node_rank=0 ../src/cryofifa_ddpfix.py \
  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 70 --minibatch-size 32 \
  --mask --input_radius 0.4 --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN \
  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1 \
  --particles_meta /sddn/kongfang_work/cryoFIFA/data1/ribo_snr05.star --using_shift_coords \
  --data_path /sddn/kongfang_work/YANG/Projects/CS-yang/ --decoder_add_layers 2 \
  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 32 --mask_epoch 2 \
  --fourier --imgsize 128 --lowpass --preprocessing_output_channels 2 --skip_read_pose \
  --lmdb_path /data/lmdb/ribo_snr05 --float16 --reduce_input --start_resolution 80 --sym \
  --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8 --noise_radius 0.4 \
  --batch_size 32 --preprocess --end_resolution 8.0 --free_gpu_memory 4 --add_distrubance \
  --distrubance_ratio 0.4 --distrubance_epoch 40 --norm6d --uniform --uniform_epoch 45 \
  --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --channels_last --fast_ddp \
  --grad_checkpoint --volume_prior exponential --volume_prior_weight 0.05 --volume_prior_beta 6 \
  --volume_prior_start_radius 0.4 --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore \
  --seed 0 --save_pose --save_checkpoint --checkpoint_interval 10 --checkpoint_keep_all ${EXTRA} ${MODEFLAGS}
