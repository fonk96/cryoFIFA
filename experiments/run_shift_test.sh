#!/bin/bash
# Shift-convention diagnostic: resume the box-128 ribosome (SNR 0.5) checkpoint and run --shift_test --
# renders synthetic no-CTF particles at known pose+shift with the model's own decoder, encodes them back,
# reports copy-0 predicted-vs-known shift (corr + linear map) + rotation sanity. Result -> /tmp/shift_test.npz.
for c in /sddn/kongfang_work/pkgs/miniconda3 ~/miniconda3; do [ -f $c/etc/profile.d/conda.sh ] && source $c/etc/profile.d/conda.sh && break; done
conda activate cryofifa
Date=2026_Jul_06
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ckpt=results/${Date}/${1:-ribo_snr05}/checkpoint.pt
torchrun --master_port 12177 --nproc_per_node 8 --nnode 1 --node_rank=0 ../src/cryofifa_ddpfix.py \
  --shift_test --resume ${ckpt} \
  --decoder_layers 2 --decoder_hidden_dim 256 --render_path results/${Date}/shift_test/ --num-epochs 1 \
  --minibatch-size 32 --mask --input_radius 0.4 --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN \
  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 100 --split_rate 1 \
  --particles_meta /sddn/kongfang_work/cryoFIFA/data1/ribo_snr05.star --using_shift_coords \
  --data_path /sddn/kongfang_work/YANG/Projects/CS-yang/ --decoder_add_layers 2 \
  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 32 --mask_epoch 2 \
  --fourier --imgsize 128 --lowpass --preprocessing_output_channels 2 --skip_read_pose \
  --lmdb_path /data/lmdb/ribo_snr05 --float16 --reduce_input --start_resolution 80 --sym \
  --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8 --noise_radius 0.4 \
  --batch_size 32 --preprocess --end_resolution 8.0 --free_gpu_memory 4 --norm6d --uniform \
  --channels_last --fast_ddp --grad_checkpoint --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore
