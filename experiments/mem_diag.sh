Date=2026_Jul_06
source ../configs/paths
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# Fast refine-phase MEMORY diagnostic: compressed marching (mask_steps 6, mask_epoch 1) so radius
# crosses downsample_radius(64) ~ep5 -> box-228 local-refine engages; profile MEMORY at ep6.
# Same memory-relevant flags as the real run (sym 8, box228, channels_last, grad_checkpoint).
save_dir=results/${Date}/mem_diag/
torchrun  --master_port 12142 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 8 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 100 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 6 --mask_epoch 1 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 3.0  --free_gpu_memory 8 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 5 --norm6d --uniform --uniform_epoch 8 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --channels_last --grad_checkpoint --fast_ddp --profile --profile_epoch 6 --profile_iters 8 --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore
