TAG=$1
Date=2026_Jul_06
if [ ! -d results/${Date} ] ; then mkdir -p results/${Date}; chmod -R 777 results/${Date}; fi
source ../configs/paths
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# SNR sweep training (2026-07-13): box-128 ribosome simulation, NO downsample_train, 70 epochs,
# mask_steps 32 x mask_epoch 2, clean (no --save_pose) + checkpoint -> best-of-8 pose_infer afterwards.
save_dir=results/${Date}/ribo_snr${TAG}/
torchrun  --master_port 12160 --nproc_per_node 8 --nnode 1 --node_rank=0  ../src/cryofifa_ddpfix.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 70 --minibatch-size 32 --mask --input_radius 0.4  --reconstruct_radius 0.38 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta /sddn/kongfang_work/cryoFIFA/data1/ribo_snr${TAG}.star --using_shift_coords --data_path /sddn/kongfang_work/YANG/Projects/CS-yang/ --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 32 --mask_epoch 2 --fourier  --imgsize 128  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data/lmdb/ribo_snr${TAG} --float16 --reduce_input --start_resolution 80 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.4 --batch_size 32 --preprocess  --end_resolution 8.0  --free_gpu_memory 4 --add_distrubance --distrubance_ratio 0.4 --distrubance_epoch 40 --norm6d --uniform --uniform_epoch 45 --lr_schedule cosine --lr_warmup --lr_warmup_epochs 5 --min_lr 1e-5 --channels_last --fast_ddp --grad_checkpoint --volume_prior exponential --volume_prior_weight 0.05 --volume_prior_beta 6 --volume_prior_start_radius 0.4 --soft_mask --soft_mask_width 0.03 --enforce_non_neg --norm_zscore --seed 0 --save_checkpoint --checkpoint_interval 10
