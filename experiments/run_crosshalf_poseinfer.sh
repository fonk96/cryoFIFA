#!/bin/bash
# CROSS-HALF GENERALIZATION (2026-07-14): the HALF1-trained checkpoint predicts poses on the UNSEEN
# HALF2 particle set in a single amortized forward pass (encoder->pose; half1 decoder only for best-of-8
# in-plane selection). --skip_read_pose so the half2 cryoSPARC poses are NEVER used for prediction.
# Architecture = the norefine ablation arm (single full-res encoder, box 228, ms32 me3, end 3.5).
Date=2026_Jul_06
source ../configs/paths
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ckpt=results/${Date}/B2_ablation_norefine/checkpoint.pt          # trained on B2_act_half1
save_dir=results/${Date}/B2_crosshalf_h1model_h2data_poses/
mkdir -p ${save_dir}
torchrun --master_port 12172 --nproc_per_node 8 --nnode 1 --node_rank=0 ../src/cryofifa_ddpfix.py \
  --pose_infer --resume ${ckpt} \
  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 200 --minibatch-size 32 \
  --mask --input_radius 0.4 --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN \
  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1 \
  --particles_meta ${B2_act_half2} --using_shift_coords --data_path /data/cryoFIFA_data/ \
  --decoder_add_layers 2 --encoder_activation SIREN --last_activation SIREN --lowpassmask \
  --mask_steps 32 --mask_epoch 3 --fourier --imgsize 228 --lowpass --preprocessing_output_channels 2 \
  --skip_read_pose --lmdb_path /data/lmdb/B2_act_half2 --float16 --reduce_input --start_resolution 40 \
  --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8 --noise_radius 0.4 \
  --batch_size 32 --preprocess --end_resolution 3.5 --free_gpu_memory 8 --add_distrubance \
  --distrubance_ratio 0.4 --distrubance_epoch 60 --norm6d --uniform --uniform_epoch 90 \
  --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --fast_ddp --channels_last \
  --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore
