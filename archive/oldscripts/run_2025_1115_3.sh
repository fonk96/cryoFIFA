
Date=2025_Nov_11
if [ ! -d results/${Date} ] ; then
mkdir -p results/${Date}
chmod -R 777 results/${Date}
fi 
 #export CUDA_VISIBLE_DEVICES=0,1,2,3

source ./paths
#save_dir=results/${Date}/save_path/


export OMP_NUM_THREADS=1



# Great success, no fade in needed, only adjust lr of the local module according to the local_refine_scale
#    save_dir=results/${Date}/ribo_snr050_downtrain_ls03_slow/
# torchrun  --master_port 12139 --nproc_per_node 8 --nnode 1 --node_rank=0  cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 100 --minibatch-size 32 --mask --input_radius 0.41  --reconstruct_radius 0.41 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${ribo_snr050} --using_shift_coords --data_path ${dgx_path} --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 96 --mask_epoch 1 --fourier  --imgsize 128  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data2/kongfang/data/lmdb/ribo_snr050 --float16 --reduce_input --start_resolution 80 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 8  --noise_radius 0.41 --batch_size 32 --preprocess  --end_resolution 7 --free_gpu_memory 8 --downsample_train --downsample_radius 32  --fade_in_epochs 0  --local_refine_scale 0.3


#This should be used for publication
#  export CUDA_VISIBLE_DEVICES=4,5,6,7
# save_dir=results/${Date}/ribo_b256_snr030_1114_ls01_run2_4gpu/
# torchrun  --master_port 12139 --nproc_per_node 4 --nnode 1 --node_rank=0  cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 200 --minibatch-size 64 --mask --input_radius 0.41  --reconstruct_radius 0.41 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${ribo_b256_snr030} --using_shift_coords --data_path ${dgx_path} --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 200 --mask_epoch 1 --fourier  --imgsize 256  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /data2/kongfang/data/lmdb/ribo_b256_snr030 --float16 --reduce_input --start_resolution 120 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 4  --noise_radius 0.41 --batch_size 64 --preprocess  --end_resolution 3.8  --free_gpu_memory 8 --save_pose --output_pose_file ribo_256_local --downsample_train --downsample_radius 48 --local_refine_scale 0.1

   export CUDA_VISIBLE_DEVICES=4,5,6,7


#    save_dir=results/${Date}/ribo_snr050_downtrain16_ls02/
# torchrun  --master_port 12139 --nproc_per_node 4 --nnode 1 --node_rank=0  cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 100 --minibatch-size 64 --mask --input_radius 0.4  --reconstruct_radius 0.4 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${ribo_snr050} --using_shift_coords --data_path ${dgx_path} --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 96 --mask_epoch 1 --fourier  --imgsize 128  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /sddn/kongfang_work/cryoFIFA/data/lmdb/ribo_snr050 --float16 --reduce_input --start_resolution 80 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 4  --noise_radius 0.4 --batch_size 64 --preprocess  --end_resolution 7 --free_gpu_memory 8 --downsample_train --downsample_radius 16  --fade_in_epochs 0  --local_refine_scale 0.2 --save_pose --output_pose_file ribosnr50_down16

#    save_dir=results/${Date}/ribo_snr050_downtrain32_ls02/
# torchrun  --master_port 12139 --nproc_per_node 4 --nnode 1 --node_rank=0  cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 100 --minibatch-size 64 --mask --input_radius 0.4  --reconstruct_radius 0.4 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${ribo_snr050} --using_shift_coords --data_path ${dgx_path} --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 96 --mask_epoch 1 --fourier  --imgsize 128  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /sddn/kongfang_work/cryoFIFA/data/lmdb/ribo_snr050 --float16 --reduce_input --start_resolution 80 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 4  --noise_radius 0.4 --batch_size 64 --preprocess  --end_resolution 7 --free_gpu_memory 8 --downsample_train --downsample_radius 32  --fade_in_epochs 0  --local_refine_scale 0.2 --save_pose --output_pose_file ribosnr50_down32

   save_dir=results/${Date}/ribo_snr050_downtrain48_ls02/
torchrun  --master_port 12139 --nproc_per_node 4 --nnode 1 --node_rank=0  cryofifa.py  --decoder_layers 2 --decoder_hidden_dim 256 --render_path ${save_dir} --num-epochs 100 --minibatch-size 64 --mask --input_radius 0.4  --reconstruct_radius 0.4 --w0 48 --w 32 --activation SIREN  --encoder_backbone resnet --encoder_basename resnet18 --save-interval 10 --split_rate 1  --particles_meta ${ribo_snr050} --using_shift_coords --data_path ${dgx_path} --decoder_add_layers 2  --encoder_activation SIREN --last_activation SIREN --lowpassmask --mask_steps 96 --mask_epoch 1 --fourier  --imgsize 128  --lowpass   --preprocessing_output_channels 2 --skip_read_pose  --lmdb_path /sddn/kongfang_work/cryoFIFA/data/lmdb/ribo_snr050 --float16 --reduce_input --start_resolution 80 --sym --encoder_hidden_dim 256 --sym_factor 8 --rotation6d --world_size 4  --noise_radius 0.4 --batch_size 64 --preprocess  --end_resolution 7 --free_gpu_memory 8 --downsample_train --downsample_radius 48  --fade_in_epochs 0  --local_refine_scale 0.2 --save_pose --output_pose_file ribosnr50_down48