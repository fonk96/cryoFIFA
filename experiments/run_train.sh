Date=2026_Jul_06
if [ ! -d results/${Date} ] ; then mkdir -p results/${Date}; chmod -R 777 results/${Date}; fi
source ../configs/paths
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# STAGE 1 - CLEAN TRAINING (2026-07-11): the proven warm-lr config, NO --save_pose (so training is never
# perturbed by the with_name data-pipeline change), --seed 0 for a fixed roll, --checkpoint_interval 10 so
# the trained encoder is saved to checkpoint.pt. Poses are exported SEPARATELY afterward via run_pose_infer.sh
# (--pose_infer --resume checkpoint.pt), which forwards the frozen encoder only and never touches training.
save_dir=results/${Date}/B2_h1_train2/
torchrun  --master_port 12151 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa_ddpfix.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 130 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${B2_act_half1} --using_shift_coords --data_path /data/cryoFIFA_data/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 64 --mask_epoch 2 --fourier  --imgsize 228  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/B2_act_half1 --float16 --reduce_input --start_resolution 30 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 2.0  --free_gpu_memory 4 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 60 --norm6d --uniform --uniform_epoch 70 --downsample_train --downsample_radius 64 --local_refine_scale 0.3 --fade_in_epochs 1 --downsample_soft_crop 0.15 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --channels_last --fast_ddp --grad_checkpoint --volume_prior exponential --volume_prior_weight 0.05 --volume_prior_beta 6 --volume_prior_start_radius 0.4 --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore --seed 0 --save_checkpoint --checkpoint_interval 10
