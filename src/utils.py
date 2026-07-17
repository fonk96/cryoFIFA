import torch
import numpy as np
from untiled import Dataloader as DL
from swd import swd

import torch.nn.functional as F
import pytorch3d.transforms as transforms
from torch.autograd import Variable
import torchvision
import sigproc
import torch_sigproc
from glob import glob
from tqdm import *
def one_hot_encoding(input_len):
  coding=torch.zeros(input_len,input_len)
  for i in range(input_len):
    coding[i,i]=1.
  return coding

def get_parser():
    import argparse

    parser = argparse.ArgumentParser('Train spatial-VAE on particle datasets')

    parser.add_argument('--data_path', default=None)
    parser.add_argument('--particles_meta', default=None)

    parser.add_argument("--local-rank", type=int,default=-1)

    parser.add_argument('--ctf-train', help='path to CTF parameters for training images')
    parser.add_argument('--ctf-test', help='path to CTF parameters for testing images')
    parser.add_argument('--scale', default=1, type=float, help='used to scale the ang/pix if images were binned (default: 1)')
    # parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default: 2)')

    parser.add_argument('--decoder_hidden_dim', type=int, default=500, help='')
    parser.add_argument('--decoder_layers', type=int, default=2, help='number of hidden layers for generator (default: 2)')
    parser.add_argument('--encoder_hidden_dim', type=int, default=500, help='')
    parser.add_argument('--encoder_layers', type=int, default=2, help='number of hidden layers for inference net (default: 2)')
    
    parser.add_argument('-a', '--activation', choices=['tanh', 'relu','SIREN'], default='tanh', help='activation function (default: tanh)')
    parser.add_argument('--resid', action='store_true', help='use residual connections in networks')
    parser.add_argument('--bilinear', action='store_true', help='use bilinear layer between coordinate and latent in spatial generator network')
    parser.add_argument('--fix_rotate', action='store_true', default=False, help='do not perform rotation inference but using input pose')
    parser.add_argument('--fix_translate', action='store_true', default=False, help='do not perform translation inference but using input pose')
    parser.add_argument('--dx_scale', type=float, default=0.1, help='standard deviation of translation latent variables (default: 0.1)')
    parser.add_argument('--x_scale', type=float, default=1.)    
    parser.add_argument('--theta-prior', type=float, default=np.pi, help='standard deviation on rotation prior (default: pi)')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--lr_schedule', type=str, default='none', choices=['none','cosine'], help="Per-epoch LR schedule. 'cosine' decays from --learning_rate to --min_lr over the run (reaching min only at num-epochs, so high-res shells that open late are still learned). 'none' = fixed LR (original).")
    parser.add_argument('--lr_warmup', action='store_true', default=False, help='Linearly warm the LR from --min_lr up to --learning_rate over the first --lr_warmup_epochs, before the cosine decay.')
    parser.add_argument('--lr_warmup_epochs', type=int, default=5, help='Number of warmup epochs when --lr_warmup is set.')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Floor LR for --lr_schedule cosine (and warmup start).')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--uniform_epoch', type=int, default=100)    
    parser.add_argument('--minibatch-size', type=int, default=100, help='minibatch size (default: 100)')
    parser.add_argument('--batchsize_tiny', type=int, default=8, help='minibatch size (default: 100)')    
    parser.add_argument('--batchsize_change_epoch', type=int, default=10000, help='minibatch size (default: 10000)')      
    parser.add_argument('--batchsize_final', type=int, default=4, help='minibatch size (default: 100)')    
    parser.add_argument('--batchsize_final_epoch', type=int, default=10000, help='minibatch size (default: 10000)')   

    parser.add_argument('--lock_epoch', type=int, default=10000, help='epoch to lock feature extrractor (default: 10000)')   


    parser.add_argument('--batchsize_finetune', type=int, default=8, help='minibatch size (default: 100)')    
    parser.add_argument('--batchsize_finetune_epoch', type=int, default=10000, help='minibatch size (default: 10000)')  

    parser.add_argument('--z-delay', type=int, default=0, help='delay using unstructured latent variables for this many training epochs (default: 0)')
    parser.add_argument('--normalize', action='store_true', help='normalize the images before training')
    parser.add_argument('-c', '--crop', type=int, default=-1, help='crop particles down to this size (default: -1 = unused)')
    # parser.add_argument('--save-prefix', default='',help='path prefix to save models (optional)')
    parser.add_argument('--save-interval', default=10, type=int, help='save frequency in epochs (default: 10)')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of training epochs (default: 100)')
    # parser.add_argument('--temp_dir',type=str)
    parser.add_argument('--em',action='store_true',default=False)
    parser.add_argument('--L1',action='store_true',default=False)
    parser.add_argument('--ignore_DC_component',action='store_true',default=False)
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    parser.add_argument('--no-preload', action='store_true', help='do not preload data into GPU RAM')
    parser.add_argument('--mask', action='store_true', help='apply a circular mask to the images')
    parser.add_argument('--save_pose', action='store_true', default=False, help='save pose file')
    parser.add_argument('--output_pose_file',type=str,default='temp_pose')

    parser.add_argument('--input_radius',type=float,default=0.4)
    parser.add_argument('--reconstruct_radius',type=float,default=0.4)

    parser.add_argument('--DW_factor',type=float,default=2.)
    parser.add_argument('--uniform',action='store_true',default=False)
    parser.add_argument('--DW',action='store_true',default=False)    
    parser.add_argument('--exp_mask',action='store_true',default=False)
    parser.add_argument('--butterworth',action='store_true',default=False)
    parser.add_argument('--fixpose',action='store_true',default=False)
    parser.add_argument('--coordinate_norm',action='store_true',default=False)
    parser.add_argument('--SIREN',action='store_true',default=False)
    parser.add_argument('--w0',type=float,default=40.)
    parser.add_argument('--w',type=float,default=30.)
    parser.add_argument('--split_rate',type=float,default=0.9)
    parser.add_argument('--em_start_epoch',type=int,default=300)

    parser.add_argument('--ori_angpix',type=float,default=1.0825)

    parser.add_argument('--render_imgsize',type=int,default=128)
    # parser.add_argument('--crop_imgsize',type=int,default=48)
    parser.add_argument('--render_number',type=int,default=1)
    parser.add_argument('--train_imgsize',type=int,default=128)

    parser.add_argument('--render_path', default='',help='path  to save results')

    parser.add_argument('--fourier',action='store_true',default=False)

    parser.add_argument('--real',action='store_true',default=False)


    parser.add_argument('--fourier_iter',type=int,default=100)
    parser.add_argument('--T',type=float,default=0.5)

    parser.add_argument('--double_shift',action='store_true',default=False)
    parser.add_argument('--skip_shift',action='store_true',default=False)
    parser.add_argument('--skip_read_pose',action='store_true',default=False)
    parser.add_argument('--render_mask',action='store_true',default=False)
    parser.add_argument('--norm6d',action='store_true',default=False)
    parser.add_argument('--grad',action='store_true',default=False)


    parser.add_argument('--render_radius',type=float,default=0.375)

    parser.add_argument('--theta_to_matrix',action='store_true',default=False)
    parser.add_argument('--lowpass_iter',type=int,default=20)

    parser.add_argument('--pose_type',type=int,default=3)
    parser.add_argument('--float16',action='store_true',default=False)
    parser.add_argument('--lowpass',action='store_true',default=False)

    parser.add_argument('--res',action='store_true',default=False)
    parser.add_argument('--single',action='store_true',default=False)
    parser.add_argument('--vae',action='store_true',default=False)

    parser.add_argument('--reg',action='store_true',default=False)
    parser.add_argument('--template_path',type=str,default='')
    parser.add_argument('--use_projections',action='store_true',default=False)
    parser.add_argument('--limit_rotation',action='store_true',default=False)

    parser.add_argument('--af2_quater',action='store_true',default=False)
    parser.add_argument('--rotate_quater',action='store_true',default=False)
    parser.add_argument('--rotation6d',action='store_true',default=False)


    parser.add_argument('--lowpassmask',action='store_true',default=False)
    parser.add_argument('--preprocessing_output_channels', type=int, default=2,help='Number of octaves in the gaussian pyramid.')

    parser.add_argument('--num_encoding', type=int, default=6,help='Positonal Encoding num')

    parser.add_argument('--euler',action='store_true',default=False)

    parser.add_argument('--encoder_add_layers',type=int,default=2)
    parser.add_argument('--decoder_add_layers',type=int,default=2)   
    parser.add_argument('--skip_ctf',action='store_true',default=False)
    parser.add_argument('--addition_path',type=str,default='')

    parser.add_argument('--fit_noise',action='store_true',default=False)
    parser.add_argument('--SNR',type=float,default=0.1)
    parser.add_argument('--sym',action='store_true',default=False)
    parser.add_argument('--ones',action='store_true',default=False)
    parser.add_argument('--norm',action='store_true',default=False)
    parser.add_argument('--standalone',action='store_true',default=False)
    parser.add_argument('--using_shift_coords',action='store_true',default=False)
    parser.add_argument('--volume_path',type=str,default='')
    parser.add_argument('--simu_particles',type=int,default=2000)
    parser.add_argument('--auto_decoder',action='store_true',default=False)
    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--decoder_only',action='store_true',default=False)
    # parser.add_argument('--save_simulation',action='store_true',default=False)
    parser.add_argument('--simulation_name',type=str,default='')
    parser.add_argument('--use_symmetrizer',action='store_true',default=False)
    parser.add_argument('--freeze_encoder',action='store_true',default=False)

    parser.add_argument('--ctf_file',type=str,default='')

    #parser.add_argument('--L1_loss',action='store_true',default=False)
    parser.add_argument('--pixel_loss',action='store_true',default=False)
    parser.add_argument('--initial_batchsize',type=int,default=32)
    parser.add_argument('--initial_iter',type=int,default=400)
    parser.add_argument('--positional_encoding',action='store_true',default=False)
    parser.add_argument('--sym_factor',type=int,default=2)

    parser.add_argument('--shared_layers',type=int,default=2)
    parser.add_argument('--encoder_activation',type=str,default='tanh')
    parser.add_argument('--last_activation',type=str,default='tanh')
    parser.add_argument('--noise_radius',type=float,default=0.45)
    parser.add_argument('--sub_dicoder_layers',type=int,default=2)
    parser.add_argument('--fast_subset_iter',type=int,default=40)
    parser.add_argument('--fast_subset_rate',type=float,default=0.2)
    parser.add_argument('--noise_KL',action='store_true',default=False)
    parser.add_argument('--score_only',action='store_true',default=False)
    parser.add_argument('--include_input',action='store_true',default=False)
    parser.add_argument('--amsgrad',action='store_true',default=False)
    parser.add_argument('--rot_manner',action='store_true',default=False)
    parser.add_argument('--optimizer',type=str,default='adam')
    parser.add_argument('--encoder_backbone',type=str,default='resnet')
    parser.add_argument('--encoder_basename',type=str,default='resnet18')
    # encoder feature-map capture (interpretability): every --featmap_interval epochs, dump the resnet
    # conv-stage activations for --featmap_n FIXED particles. rank0-only, eval/no_grad on a detached
    # clone -> zero effect on training math. Off by default.
    parser.add_argument('--dump_featmaps',action='store_true',default=False)
    parser.add_argument('--featmap_interval',type=int,default=10)
    parser.add_argument('--featmap_n',type=int,default=10)
    # shift-convention diagnostic: render synthetic no-CTF particles with the model's OWN decoder at
    # KNOWN poses + KNOWN in-plane shifts, encode them back, and log predicted vs known shift (copy 0).
    parser.add_argument('--shift_test',action='store_true',default=False)
    # cross-correlation shift refinement in training: once structure exists (epoch>=cc_shift_start_epoch),
    # use the CC-aligned shift for the reconstruction loss AND supervise the encoder shift head toward it
    # (so the head learns an accurate, amortized shift instead of the weak/2x-underscaled prediction).
    parser.add_argument('--cc_shift',action='store_true',default=False)
    parser.add_argument('--cc_shift_start_epoch',type=int,default=40)
    parser.add_argument('--cc_shift_weight',type=float,default=0.1)
    # post-hoc CC shift at INFERENCE (--pose_infer): export the cross-correlation shift of the winning
    # copy (accurate, decoupled from ab-initio) instead of the weak head shift. Safe -- no training change.
    parser.add_argument('--cc_export',action='store_true',default=False)
    # per-particle loss export (with --pose_infer): write <name loss> per particle (for loss-vs-SNR /
    # contribution-tracking diagnostics on mixed-SNR data). And keep per-epoch checkpoints for that tracking.
    parser.add_argument('--loss_export',action='store_true',default=False)
    # per-particle best-of-8 CONFIDENCE export (with --pose_infer): from the 8 in-plane hypothesis
    # distances -> conf_infer_rank*.txt (name d1 d2 dmean dstd wmax entropy). Self-normalized within a
    # particle (SNR-independent) -> a GT-free filter that should beat raw loss for picking well-posed particles.
    parser.add_argument('--conf_export',action='store_true',default=False)
    parser.add_argument('--checkpoint_keep_all',action='store_true',default=False)
    parser.add_argument('--mask_steps',type=int,default=3)
    parser.add_argument('--mask_epoch',type=int,default=10)
    parser.add_argument('--port',type=str,default='12306')
    parser.add_argument('--world_size',type=int,default=4)  # 单机多卡


    parser.add_argument('--rot_loss',action='store_true',default=False)


    parser.add_argument('--kernel_size',type=int,default=11)

    parser.add_argument('--kernel_variance',type=float,default=0.01)
    parser.add_argument('--out_device',type=int,default=2)
    parser.add_argument('--resdecoder',action='store_true',default=False)
    parser.add_argument('--noise_estimation_images',type=int,default=100)
    parser.add_argument('--gaussian',action='store_true',default=False)
    parser.add_argument('--reg_dim',type=int,default=4)
    parser.add_argument('--shift_ori',action='store_true',default=False)

    parser.add_argument('--encoder_loss',action='store_true',default=False)
    parser.add_argument('--rescale',type=float,default=2.)

    parser.add_argument('--superes',action='store_true',default=False)

    parser.add_argument('--downsample_train',action='store_true',default=False)
    parser.add_argument('--down_factor',type=int,default=2)
    parser.add_argument('--swapxy',action='store_true',default=False)
    parser.add_argument('--flipx',action='store_true',default=False)
    parser.add_argument('--flipy',action='store_true',default=False)
    # parser.add_argument('--render_size',type=int,default=256)
    # parser.add_argument('--ori_size',type=int,default=192)
    parser.add_argument('--update_shift',action='store_true',default=False)
    parser.add_argument('--output_dir',type=str,default='')

    parser.add_argument('--upsample_infer',action='store_true',default=False)
    parser.add_argument('--W_loss',action='store_true',default=False)
    parser.add_argument('--run2d',action='store_true',default=False)
    parser.add_argument('--local_refine',action='store_true',default=False)
    parser.add_argument('--sampling_method',type=str,default='avgpooling')
    parser.add_argument('--input_volume',type=str,default='')
    parser.add_argument('--input_folder',type=str,default='refs/')
    parser.add_argument('--out_dim',type=int,default=2)
    parser.add_argument('--latent_code_dim_2d',type=int,default=0)
    parser.add_argument('--latent_code_dim_3d',type=int,default=0)
    parser.add_argument('--pretrain_num_epochs',type=int,default=0)
    parser.add_argument('--pretrain',action='store_true',default=False)
    parser.add_argument('--trim',action='store_true',default=False)
    parser.add_argument('--direct',action='store_true',default=False)
    parser.add_argument('--simulate_pretrain',action='store_true',default=False)
    parser.add_argument('--pretrain_mini_batch_size',type=int,default=64)
    parser.add_argument('--raw_input',action='store_true',default=False)
    parser.add_argument('--flatten',action='store_true',default=False)
    parser.add_argument('--imgsize',type=int,default=128)
    parser.add_argument('--decoder_freeze_epoch',type=int,default=0)

    parser.add_argument('--simulate_num',type=int,default=10000)

    parser.add_argument('--factor',type=float,default=1.)
    parser.add_argument('--start_resolution',type=float,default=30.)    
    parser.add_argument('--end_resolution',type=float,default=6.)       
    # parser.add_argument('--lmdb',type=str,default='')
    parser.add_argument('--lmdb_path',type=str,default='')
    parser.add_argument('--load_data_before_train',action='store_true',default=False)
    parser.add_argument('--reduce_input',action='store_true',default=False)
    parser.add_argument('--consistance_loss',action='store_true',default=False)
    parser.add_argument('--calc_noise',action='store_true',default=False)
    parser.add_argument('--force_full_std',action='store_true',default=False)

    parser.add_argument('--hard_topk',action='store_true',default=False)
    parser.add_argument('--soft_topk',action='store_true',default=False)
    parser.add_argument('--topk_ratio',type=float,default=0.8)   

    parser.add_argument('--topk_start_epoch',type=int,default=0)  

    parser.add_argument('--topk_end_epoch',type=int,default=0) 
    parser.add_argument('--topk_start_ratio',type=float,default=1.)   
    parser.add_argument('--topk_end_ratio',type=float,default=1.) 

    parser.add_argument('--double',action='store_true',default=False)    
    parser.add_argument('--particles_meta_double',type=str,default='')    
    parser.add_argument('--invert_ed',action='store_true',default=False)         
    parser.add_argument('--mode',type=str,default='3d')    


    #----------DOWN SAMPLE TRAININNG OPTIONS-----------

    parser.add_argument('--downsample_radius',type=int,default=32)
    parser.add_argument('--downsample_soft_crop',type=float,default=0.0,help='Apodize the downsample_input Fourier crop with a raised-cosine window tapering over this outer fraction of the crop (e.g. 0.15). 0 = original hard box crop (sinc ringing). Removes the ringing the double-encoder global branch was fed.')
    parser.add_argument('--downsample_train_epoch',type=int,default=0)
    parser.add_argument('--coarse_only',action='store_true',default=False,help='downsample_train but NEVER engage the local full-res encoder: pose is predicted only from the low-res coarse input for ALL epochs (frequency marching still drives the decoder/loss). Skips the local encoder forward entirely -> cheaper. Result-preserving when off.')
    parser.add_argument('--translation_restrain',action='store_true',default=False)

    parser.add_argument('--translation_restrain_epoch',type=int,default=0)  
        
    parser.add_argument('--fade_in_epochs',type=int,default=0) 
    parser.add_argument('--local_only',action='store_true',default=False) 
    parser.add_argument('--random_input',action='store_true',default=False)     
    parser.add_argument('--random_noise',action='store_true',default=False)    
    #-------------------------------------------------
    parser.add_argument('--sigma_level',type=float,default=3.)  
    parser.add_argument('--trunk_size',type=int,default=8) 
    parser.add_argument('--free_gpu_memory',type=float,default=2.) 

    parser.add_argument('--random_augmentation',action='store_true',default=False)    
    #------------MASK OPTIONS-------------#


    parser.add_argument('--preprocess',action='store_true',default=False)
    parser.add_argument('--lowpass_input',action='store_true',default=False)
    parser.add_argument('--mask_lowpass_threshold',type=float,default=128,help='Volume values will range from 0-255, so 0.2 in cryoSPARC may be 255*0.2 here. ')    

    parser.add_argument('--mask_dilation_width',type=int,default=3)    
    parser.add_argument('--mask_padding_width',type=int,default=12)   



    parser.add_argument('--mask_start_epoch',type=float,default=1000)  

    parser.add_argument('--mask_lowpass',type=float,default=15)  

    parser.add_argument('--mask_lowpass_type',type=str,default='butterworth')

    parser.add_argument('--mask_lowpass_order',type=int,default=6) 
    parser.add_argument('--write_device',type=int,default=1) 
    #--------------------------------------#

    parser.add_argument('--local_refine_scale',type=float,default=0.1)
    parser.add_argument('--log_refine_delta',action='store_true',default=False,help='Log the local refiner angular/shift correction (coarse pose -> refined pose) per minibatch to refine_delta.txt. Opt-in, result-preserving.')
    parser.add_argument('--global_bins',type=float,default=18)
    parser.add_argument('--use_KAN',action='store_true',default=False)  

    parser.add_argument('--global_smooth',action='store_true',default=False)
    parser.add_argument('--add_distrubance',action='store_true',default=False)    
    parser.add_argument('--distrubance_ratio',type=float,default=0.1)     
    parser.add_argument('--distrubance_end_ratio',type=float,default=0.01)   
    parser.add_argument('--distrubance_epoch',type=int,default=30)       
    parser.add_argument('--gaussian_distrubance',action='store_true',default=False)   
    parser.add_argument('--using_signal_mask',action='store_true',default=False)     
    #------------------------------
    parser.add_argument('--use_batchnorm',action='store_true',default=False)
    parser.add_argument('--downsample_denoise',action='store_true',default=False)    
    parser.add_argument('--fourier_downsample',action='store_true',default=False)       

    parser.add_argument('--cross_validation',action='store_true',default=False)   

    parser.add_argument('--denoise',action='store_true',default=False)    

    parser.add_argument('--diffusion',action='store_true',default=False)   
    parser.add_argument('--diffusion_dim',type=int,default=8)      
    #-------------------------------------
    parser.add_argument('--load_checkpoint',action='store_true',default=False)
    parser.add_argument('--save_checkpoint',action='store_true',default=False)
    parser.add_argument('--test_only',action='store_true',default=False)
    parser.add_argument('--checkpoint_path',type=str,default='')
    parser.add_argument('--resume',type=str,default='',help='Path to a checkpoint.pt to resume {models,optimizers,scaler,epoch} from. Empty = fresh start.')
    parser.add_argument('--checkpoint_interval',type=int,default=0,help='Save a checkpoint every N epochs. 0 = use --save-interval (i.e. save whenever a volume is rendered).')
    parser.add_argument('--seed',type=int,default=-1,help='Fixed RNG seed (>=0) for reproducible ab-initio runs (same basin each run). -1 = seedless (default).')
    parser.add_argument('--pose_infer',action='store_true',default=False,help='Stage-2 pose inference: with --resume <checkpoint>, forward-only export of the frozen encoder pose per particle (no sym/disturbance/training), then exit. Decoupled from training so it never perturbs it.')

    #-------------VOLUME REGULARIZATION / OUTPUT OPTIONS (added upgrade)-------------
    # 2. Soft-edge (raised-cosine) masks instead of hard binary spheres -> kill Gibbs ringing.
    parser.add_argument('--soft_mask',action='store_true',default=False,help='Use raised-cosine edges on the reconstruction support masks and the render Fourier low-pass, instead of hard binary cuts.')
    parser.add_argument('--soft_mask_width',type=float,default=0.03,help='Soft-edge taper width as a fraction of the box size (e.g. 0.03 = ~7px at box 228).')
    # 3. Gaussian spectral (Wiener/Tikhonov) volume prior in the loss.
    #    'exponential' is intentionally NOT the default: the decoder exp(envelope)*modulant
    #    already embeds an exponential-density prior, so a Gaussian shell prior is the
    #    complementary, non-redundant choice and lives in the same Fourier domain as the loss.
    parser.add_argument('--volume_prior',type=str,default='none',choices=['none','gaussian','exponential'],help="Volume prior added to the loss. 'gaussian' = shell-weighted L2 penalty w(k)=|k/kNyq|^power (RELION Wiener-style). 'exponential' = cryoSPARC-style exponential envelope penalty w(k)=exp(beta*|k/kNyq|)-1, which shrinks the highest shells far harder (B-factor-like).")
    parser.add_argument('--volume_prior_weight',type=float,default=0.0,help='Strength of the volume prior term. Start small (e.g. 1e-3..1e-1) and raise until the high-frequency noise shelf is tamed without blurring detail.')
    parser.add_argument('--volume_prior_power',type=float,default=2.0,help='Radial-frequency exponent for the Gaussian prior weight w(k)=|k/kNyq|^power. 2.0 ~ Tikhonov; larger = penalize only the highest shells.')
    parser.add_argument('--volume_prior_beta',type=float,default=6.0,help="Steepness of the 'exponential' cryoSPARC-style prior: w(k)=exp(beta*|k/kNyq|)-1. Larger beta = the penalty concentrates ever more sharply on the top shells.")
    parser.add_argument('--volume_prior_start_radius',type=float,default=0.0,help='Fraction of Nyquist below which the volume prior is 0 (only penalize shells beyond this). 0 = penalize all non-DC shells. Set ~0.4 to leave the resolved band untouched and only damp the noise band.')
    # amortized-friendly answer to pose multimodality WITHOUT an SO(3) search:
    # soft-min (marginalized) over the in-plane sym hypotheses instead of hard best-of-N,
    # so gradients flow to all plausible copies (a probabilistic pose) rather than committing to one.
    parser.add_argument('--soft_min',action='store_true',default=False,help='Marginalize the symmetric pose loss over the sym hypotheses (softmax-weighted) instead of hard min-over-copies. Keeps amortized single-pass inference but stops the encoder committing hard to one orientation.')
    parser.add_argument('--soft_min_temp',type=float,default=0.1,help='Temperature for --soft_min, RELATIVE to the mean per-hypothesis distance. ->0 recovers hard min; larger spreads weight across hypotheses. 0.05-0.2 is a sensible range.')
    # 4. Non-negativity on the real-space density (electron density is physically >=0).
    parser.add_argument('--enforce_non_neg',action='store_true',default=False,help='Clamp the rendered real-space volume to >=0, removing the unphysical negative ringing lobes.')
    # 5. Zero-mean/unit-variance-inside-mask normalization instead of global min-max.
    parser.add_argument('--norm_zscore',action='store_true',default=False,help='Normalize the output volume to zero-mean/unit-std computed inside the reconstruction mask, instead of global min-max (which lets one hot voxel set the scale).')

    #------------- PERF: Tier 0 profiling + Tier 1 fast-DDP (added upgrade) -------------
    parser.add_argument('--profile',action='store_true',default=False,help='Tier 0: run torch.profiler over --profile_iters iterations of --profile_epoch on rank 0, print the per-op CUDA-time table and dump a chrome trace to render_path, then continue training. Use to find the real bottleneck.')
    parser.add_argument('--profile_iters',type=int,default=15,help='Number of iterations to profile when --profile is set.')
    parser.add_argument('--profile_epoch',type=int,default=0,help='Which epoch to run the profiler on.')
    parser.add_argument('--fast_ddp',action='store_true',default=False,help='Tier 1: DDP gradient_as_bucket_view=True + no_sync() during gradient accumulation (all-reduce once per optimizer step instead of every micro-step). Numerically identical, just less communication. Default off for exact reproducibility.')
    parser.add_argument('--channels_last',action='store_true',default=False,help='Tier 2: run the ResNet encoder in channels_last (NHWC) memory format so V100 conv tensor cores are faster. Numerically identical (layout only). Targets the ~36%% conv time.')
    parser.add_argument('--grad_checkpoint',action='store_true',default=False,help='Activation (gradient) checkpointing on the ResNet encoder: recompute activations in backward instead of storing them -> big cut in encoder-activation memory (~25-30%% more encoder compute). Frees memory so the dynamic minibatch can stay large. NOTE: not bit-identical because BN running stats are recomputed in backward.')
    parser.add_argument('--cascade_refine',action='store_true',default=False,help='Option-B conditioned cascade: a zero-init PoseRefiner sees the render-and-compare residual (y - pred@coarse-pose) and predicts a small so(3)+shift update; the volume is then re-rendered at the refined pose. Amortised Newton step -> refines pose without per-particle optimisation. Default off (byte-identical when absent).')
    parser.add_argument('--cascade_lambda',type=float,default=0.0,help='Weight of an auxiliary loss on the COARSE render (pred@coarse-pose), so the coarse head keeps direct gradient. 0 = only the refined render is in the loss.')
    parser.add_argument('--cascade_shift_scale',type=float,default=1.0,help='Scale (in pixels) on the refiner shift residual dt.')

    #-------------------------
    args = parser.parse_args()
    return args


def num2str(input_number,output_len=5):
  out_str='0'*output_len
  return out_str[:output_len-len(str(input_number))]+str(input_number)


def data_sampling(input_tensor,sampling_method='avgpooling',down_factor=2):
  if sampling_method=='avgpooling':
    return torch.nn.AvgPool2d(down_factor)(input_tensor)
  elif sampling_method=='maxpooling':
    return torch.nn.MaxPool2d(down_factor)(input_tensor)
  elif sampling_method=='random_pick':
    return



def shift_masks(imgsize,radius,shifts):
  b=shifts.shape[0]
  masks=torch.zeros(b,imgsize,imgsize,device=shifts.device)
  for i in range(b):
    masks[i]=gen_mask(imgsize,radius,'2d',False,shifts[i])
  return masks

def trim_3d_tensor(input_tensor,axis=1,dtype=torch.float):
  # default style like
  # imgage_slices=torch.zeros(input_tensor.size(0))
  #  for i in input_tensor.size(0):
  #      image_slices =  input_tensor [i , ...]

  # so by defining new axis may get:
  # for i in input_tensor.size(1):
  #      imgae_slices = input_tensor[...,i,...]
  assert len(input_tensor.size())==4 and input_tensor.size(0)==input_tensor.size(1) and input_tensor.size(1)==input_tensor.size(2)
  new_tensor=torch.zeros(input_tensor.size(),dtype=dtype)
  for indexi in range(input_tensor.size(axis)):
    new_tensor[indexi,:,:,:]= input_tensor[:,indexi,:,:] if axis ==1 else input_tensor[:,:,indexi,:]
  return new_tensor

def one_hot_encoding(input_len,latent_code_dim):
  coding=torch.zeros(input_len,latent_code_dim)
  for i in range(input_len):
    coding[i,i]=1.
  return coding


def torch_apply_lowpass(input_tensor,steps=3,lowpass_masks=None,lowpassorder=2):
  imgsize=input_tensor.size(-1)
  input_tensor=input_tensor.view(-1,1,imgsize,imgsize)
  output_tensor=input_tensor.expand(-1,steps,imgsize,imgsize)
  output_tensor=ift2(  torch.mul(ft2(output_tensor),lowpass_masks ) ).real
  #special test
  #output_tensor[:,0,:,:]=output_tensor[:,1,:,:]
  return output_tensor
  

def apply_lowpass(input_tensor,steps=4,lowpassorder=2,include_input=False):
    imgsize=input_tensor.size(-1)
    input_tensor=input_tensor.view(-1,imgsize,imgsize)
    b=input_tensor.size(0)
    if include_input:
      output_tensor=torch.zeros(b,steps+1,imgsize,imgsize,device=input_tensor.device)  
      output_tensor[:,0,:,:]=input_tensor
    else:
      output_tensor=torch.zeros(b,steps,imgsize,imgsize,device=input_tensor.device)

    ft_input=ft2(input_tensor)

    #modified lowpass refering to cryosparc import particles
    max_radius=int(imgsize/2)
    min_radius=6
    grad=int((max_radius-min_radius)/(steps-1))

    for i in range(steps):
      # lowpass_size=int(imgsize/2**(i+1))
      # This is better if using smaller num_octaves even without ori_input ,so input lowpass still matters

      # lowpass_size=int(imgsize/2/2**(i+1))
      lowpass_size=max_radius-i*grad
      render_i=i+1 if include_input else i
      #mask =torch.from_numpy( sigproc.lowpass_mask(imgsize, 2, lowpass_size, lowpassorder=lowpassorder) ).to(input_tensor.device)
      mask=torch_sigproc.lowpass_mask(imgsize, 2, lowpass_size, lowpassorder=lowpassorder,device=input_tensor.device)

      output_tensor[:,render_i,:,:]=ift2(torch.mul(ft_input,mask)).real

    return output_tensor




def expmap(v):
    theta = v.norm(p=2, dim=-1, keepdim=True)
    # normalize K
    K = map_to_lie_algebra(v / theta)

    I = torch.eye(3, device=v.device, dtype=v.dtype)
    R = I + torch.sin(theta)[..., None]*K \
        + (1. - torch.cos(theta))[..., None]*(K@K)
    return R




def torch_lowpass(input_tensor,lowpass_ratio=8,lowpassorder=4):
    imgsize=input_tensor.size(-1)
    lowpass_size=int(imgsize/lowpass_ratio)
    mask = sigproc.lowpass_mask(imgsize, 2, lowpass_size, lowpassorder=lowpassorder)


    torch_mask=torch.from_numpy(mask).to(input_tensor.device)

    #lowpassed_tensor=torch.fft.fftn( torch.fft.ifftshift( torch.mul (torch.fft.fftshift(torch.fft.fftn(input_tensor,dim=(-2,-1)),dim=(-2,-1)) ,torch_mask),dim=(-2,-1)),dim=(-2,-1))

    lowpassed_tensor=ift2( torch.mul(ft2(input_tensor),torch_mask) )

    return lowpassed_tensor


def torch_highpass(input_tensor,highpass_ratio=3,highpassorder=10):
    imgsize=input_tensor.size(-1)
    lowpass_size=int(imgsize/highpass_ratio)
    mask = 1-sigproc.lowpass_mask(imgsize, 2, highpass_ratio, lowpassorder=highpassorder)


    torch_mask=torch.from_numpy(mask).to(input_tensor.device)

    highpassed_tensor=torch.mul(input_tensor,torch_mask)
    #highpassed_tensor=ift2( torch.mul(ft2(input_tensor),torch_mask) )

    return highpassed_tensor

def print_refresh(context):
    print(context,"\r",end="",flush=True)

def data_prior_loss(input_volume,grid,lamda=10):
    """
    This should return a probability of this input_volume
    """
    unit_neg=torch.complex(torch.tensor([0]),torch.tensor([-1])).float.to(input_volume.device)
    return lamda*(lamda-unit_neg*input_volume)


def score_complex_noise(subtracted_tensor,noise_var):
    """
    This should return a KL between Gaussian noise ~N(0,noise_var) and subtracted_tensor
    """
    torch_noise=torch.randn_like(subtracted_tensor)*torch.sqrt(noise_var)
    sub=subtracted_tensor-torch_noise
    return (sub.real**2+sub.imag**2).mean()


def score_noise(subtracted_tensor,noise_var):
    """
    This should return a KL between Gaussian noise ~N(0,noise_var) and subtracted_tensor
    """
    torch_noise=torch.randn_like(subtracted_tensor)*torch.sqrt(noise_var)

    return ((subtracted_tensor-torch_noise)**2).mean()


def noise_KL(subtracted_tensor,noise_var):
    """
    This should return a KL between Gaussian noise ~N(0,noise_var) and subtracted_tensor
    """
    #input_mean=torch.mean(subtracted_tensor)
    input_var=torch.var(subtracted_tensor)
    input_logvar=torch.log(input_var)
    #kl_div=0.5*torch.sum(input_var/noise_var +torch.square(input_mean)/noise_var -1 - torch.log(input_var/noise_var))
    #kl_div=torch.mean(torch.sum((1.0 + input_logvar - input_mean.pow(2) - input_logvar.exp()) / 2.0, dim=1), dim=0)
    kl_div = -input_logvar + 0.5*torch.log(noise_var) + input_var/2/noise_var - 0.5
    #kl_div=0.5*torch.log(input_var/noise_var)+(noise_var**2+input_mean**2)/(2*input_var)-0.5
    return kl_div

def theta_to_matrix(v1,v2=None):
    """
    make some restrains so that
    v1 should represent a unit vector
    v2 should be 3 euler angles and
    the first and the 3rd angle lie in [-pi/2,pi/2]
    and the second one lie in [0,pi]
    """
    if v2 is None:
        assert v1.shape[-1] == 6
        v2 = v1[..., 3:]
        v1 = v1[..., 0:3]
    u1 = v1
    e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    v2=v2*np.pi/2
    #v2[...,1]=v2[...,1]+np.pi/2
    new_theta=torch.cat((v1,v2),dim=-1)
    return transforms.rotation_6d_to_matrix(new_theta)


# def normalize(v, eps=1e-6):
#     norm = v.norm(dim=-1, keepdim=True)
#     return v / (norm + eps)

def sixd_to_two_orthonormal(d6: torch.Tensor, eps: float = 1e-6):
    """
    d6: (..., 6)，任意 6 维向量
    返回: (..., 3) 的 b1, (..., 3) 的 b2，两者单位且互相正交
    """
    # d6.shape = (..., 6), 我们将其拆分为两个 3D 向量
    a1 = d6[...,:3]  # 取前3个元素
    a2 = d6[...,-3:]  # 取后3个元素

    # 1) 归一化 a1（带 eps，避免除 0）
    b1 = F.normalize(a1, dim=-1, eps=eps)

    # 如果 a1 本身几乎为 0，强行指定一个方向作为 b1
    norm1 = a1.norm(dim=-1, keepdim=True)
    mask1 = norm1 < eps  # 需要兜底的地方
    if mask1.any():
        # 固定用 x 轴做兜底
        fallback_b1 = torch.zeros_like(b1)
        fallback_b1[..., 0] = 1.0
        b1 = torch.where(mask1, fallback_b1, b1)

    # 2) 去掉 a2 在 b1 上的投影，得到正交部分
    proj = (b1 * a2).sum(dim=-1, keepdim=True)
    a2_ortho = a2 - proj * b1

    # 3) 归一化 a2_ortho
    b2 = F.normalize(a2_ortho, dim=-1, eps=eps)

    # 检查 a2_ortho 是否也几乎为 0（即 a2 几乎和 b1 共线）
    norm2 = a2_ortho.norm(dim=-1, keepdim=True)
    mask2 = norm2 < eps
    if mask2.any():
        # 选一个和 b1 尽量不平行的坐标轴
        # 构造三个候选轴
        ex = torch.tensor([1.0, 0.0, 0.0], device=d6.device, dtype=d6.dtype)
        ey = torch.tensor([0.0, 1.0, 0.0], device=d6.device, dtype=d6.dtype)
        ez = torch.tensor([0.0, 0.0, 1.0], device=d6.device, dtype=d6.dtype)
        cand = torch.stack([ex, ey, ez], dim=0)   # (3, 3)

        # 计算每个样本与三个轴的点积绝对值，选最小的那个
        dots = torch.matmul(b1.unsqueeze(-2), cand.t()).abs()  # (..., 1, 3)
        idx = dots.argmin(dim=-1)  # (..., 1)

        # 根据 idx 选出 fallback 方向
        fallback_b2 = cand[idx.squeeze(-2)]       # (..., 3)

        # 再做一次正交化 & 归一化以保证和 b1 严格正交
        proj_fb = (b1 * fallback_b2).sum(dim=-1, keepdim=True)
        fb_ortho = fallback_b2 - proj_fb * b1
        fb_ortho = F.normalize(fb_ortho, dim=-1, eps=eps)

        b2 = torch.where(mask2, fb_ortho, b2)

    return torch.cat((b1,b2),dim=-1)
'''

def sixd_to_two_orthonormal(d6: torch.Tensor, eps: float = 1e-6):
    """
    d6: (..., 6)，任意 6 维向量
    返回: (..., 3) 的 b1, (..., 3) 的 b2，两者单位且互相正交
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]

    # 1) 归一化 a1（带 eps，避免除 0）
    b1 = F.normalize(a1, dim=-1, eps=eps)

    # 如果 a1 本身几乎为 0，强行指定一个方向作为 b1
    norm1 = a1.norm(dim=-1, keepdim=True)
    mask1 = norm1 < eps  # 需要兜底的地方
    if mask1.any():
        # 固定用 x 轴做兜底
        fallback_b1 = torch.zeros_like(b1)
        fallback_b1[..., 0] = 1.0
        b1 = torch.where(mask1, fallback_b1, b1)

    # 2) 去掉 a2 在 b1 上的投影，得到正交部分
    proj = (b1 * a2).sum(dim=-1, keepdim=True)
    a2_ortho = a2 - proj * b1

    # 3) 归一化 a2_ortho
    b2 = F.normalize(a2_ortho, dim=-1, eps=eps)

    # 检查 a2_ortho 是否也几乎为 0（即 a2 几乎和 b1 共线）
    norm2 = a2_ortho.norm(dim=-1, keepdim=True)
    mask2 = norm2 < eps
    if mask2.any():
        # 选一个和 b1 尽量不平行的坐标轴
        # 构造三个候选轴
        ex = torch.tensor([1.0, 0.0, 0.0], device=d6.device, dtype=d6.dtype)
        ey = torch.tensor([0.0, 1.0, 0.0], device=d6.device, dtype=d6.dtype)
        ez = torch.tensor([0.0, 0.0, 1.0], device=d6.device, dtype=d6.dtype)
        cand = torch.stack([ex, ey, ez], dim=0)   # (3, 3)

        # 计算每个样本与三个轴的点积绝对值，选最小的那个
        # b1: (..., 3)
        dots = torch.matmul(b1.unsqueeze(-2), cand.t()).abs()  # (..., 1, 3)
        idx = dots.argmin(dim=-1)  # (..., 1)

        # 根据 idx 选出 fallback 方向
        fallback_b2 = cand[idx.squeeze(-2)]       # (..., 3)

        # 再做一次正交化 & 归一化以保证和 b1 严格正交
        proj_fb = (b1 * fallback_b2).sum(dim=-1, keepdim=True)
        fb_ortho = fallback_b2 - proj_fb * b1
        fb_ortho = F.normalize(fb_ortho, dim=-1, eps=eps)

        b2 = torch.where(mask2, fb_ortho, b2)

    return b1, b2
'''

def normalize(v, eps=1e-6):
    norm = v.norm(dim=-1, keepdim=True)
    return v / (norm + eps)

def gram_schmidt_orthogonalization(v1, v2=None, eps=1e-6):
    if v2 is None:
        assert v1.shape[-1] == 6
        v2 = v1[..., 3:]
        v1 = v1[..., 0:3]  
    # Step 1: Make v1 a unit vector
    v1 = normalize(v1, eps)

    # Step 2: Remove the component of v2 in the direction of v1
    v2_orthogonal = v2 - (torch.sum(v1 * v2, dim=-1, keepdim=True) * v1)
    
    # Step 3: Normalize v2_orthogonal to make it a unit vector
    v2_orthogonal = normalize(v2_orthogonal, eps)
    
    return v1, v2_orthogonal



def image_duplicator(input_images,factor=2,rot_manner=False,degrees=[]):
    """
    input_images: [batch,channel,imgsize,imgsize]
    """
    imgsize=input_images.size(-1)
    b=input_images.size(0)
    input_images=input_images.view(b,-1,imgsize,imgsize)
    if len(degrees)>1:
      #print(degrees)
      img_out=input_images
      for di in range(len(degrees)//2-1):
        img_out=torch.cat((img_out,torchvision.transforms.functional.rotate(input_images,degrees[di])),0)
      flips=torch.flip(img_out,[-1])
      img_out=torch.cat((img_out,flips),0)        
      return img_out
    assert factor in [2, 4, 8,16], "Only 2,4,8 symmetric_factor supported"
    if factor == 2:
        if rot_manner:
          copy_img=torch.rot90(input_images,2,[-2,-1])
        else:
          copy_img=torch.flip(input_images,[-1])
        img_out = torch.cat((input_images,copy_img),0)
    elif factor==4 or factor==8 or factor==16:
        copy_img1=torch.rot90(input_images,1,[-2,-1])
        copy_img2=torch.rot90(input_images,2,[-2,-1])
        copy_img3=torch.rot90(input_images,3,[-2,-1])

        img_out=torch.cat((input_images,copy_img1,copy_img2,copy_img3),0)
        if factor==8  or factor==16:
          if rot_manner or factor==16:
            rot_45_images=torchvision.transforms.functional.rotate(img_out,45)
            img_out=torch.cat((img_out,rot_45_images),0)
            #print(img_out.size())
          if factor==16 or (factor==8 and not rot_manner):
            flips=torch.flip(img_out,[-1])
            img_out=torch.cat((img_out,flips),0)
            #print(img_out.size())            
    return img_out


def complex_loss(y_pred, y_gt):

  loss= torch.mean(((y_pred.real - y_gt.real) ** 2+(y_pred.imag - y_gt.imag) ** 2), -1)
  return loss.mean()

def torch_wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
    # Ensure that the input tensors are batched
    assert u_values.dim() == 2 and v_values.dim() == 2, "Input tensors must be 2-dimensional (batch_size, num_values)"

    batch_size, u_size = u_values.shape
    _, v_size = v_values.shape

    # Sort the values
    u_sorter = torch.argsort(u_values, dim=1)
    v_sorter = torch.argsort(v_values, dim=1)

    # Concatenate and sort all values for each batch
    all_values = torch.cat((u_values, v_values), dim=1)
    all_values, _ = torch.sort(all_values, dim=1)
    # Compute differences between successive values
    deltas = torch.diff(all_values, dim=1)

    # Get the respective positions of the values of u and v among the values of both distributions
    all_continue = all_values[:, :-1].contiguous()
    u_cdf_indices = torch.searchsorted(u_values.gather(1, u_sorter).contiguous(), all_continue, right=True)
    v_cdf_indices = torch.searchsorted(v_values.gather(1, v_sorter).contiguous(), all_continue, right=True)

    # Calculate the CDFs of u and v using their weights, if specified
    if u_weights is None:
        u_cdf = u_cdf_indices.float() / u_size
    else:
        u_sorted_cumweights = torch.cat((torch.zeros((batch_size, 1)), torch.cumsum(u_weights.gather(1, u_sorter), dim=1)), dim=1)
        u_cdf = u_sorted_cumweights.gather(1, u_cdf_indices) / u_sorted_cumweights[:, -1].unsqueeze(1)

    if v_weights is None:
        v_cdf = v_cdf_indices.float() / v_size
    else:
        v_sorted_cumweights = torch.cat((torch.zeros((batch_size, 1)), torch.cumsum(v_weights.gather(1, v_sorter), dim=1)), dim=1)
        v_cdf = v_sorted_cumweights.gather(1, v_cdf_indices) / v_sorted_cumweights[:, -1].unsqueeze(1)

    return torch.sum(torch.abs(u_cdf - v_cdf) * deltas, dim=1)

def complex_W_loss(y_pred, y_gt, sym_loss_factor=2,device=""):
    # n_pts = y_gt.shape[-1]
    y_pred = y_pred.reshape(sym_loss_factor, -1)
    y_gt = y_gt.reshape(sym_loss_factor, -1)
    distance_double=torch_wasserstein_distance(y_pred.real, y_gt.real)+torch_wasserstein_distance(y_pred.imag,y_gt.imag)
    #distance_double = torch.mean((torch.abs(y_pred.real - y_gt.real) +torch.abs(y_pred.imag - y_gt.imag) ), -1)
    min_distances, activated_paths = torch.min(distance_double, 0)

    return min_distances, activated_paths


import math
import torch


def sinkhorn_wasserstein_cost(x, y, epsilon=0.1, n_iters=50, p=2):
    """
    Differentiable Sinkhorn approximation of Wasserstein-1 (EMD) with uniform weights.

    x: [B, N, D]
    y: [B, M, D]
    epsilon: entropic regularization strength
    n_iters: Sinkhorn iterations
    p: for torch.cdist (p=2 => Euclidean)

    returns: cost [B]
    """
    assert x.dim() == 3 and y.dim() == 3
    B, N, D = x.shape
    By, M, Dy = y.shape
    assert B == By and D == Dy

    # Cost matrix: [B, N, M]
    C = torch.cdist(x, y, p=p)  # Euclidean cost; differentiable

    # Uniform marginals
    log_a = torch.full((B, N), -math.log(N), device=x.device, dtype=x.dtype)
    log_b = torch.full((B, M), -math.log(M), device=x.device, dtype=x.dtype)

    # Log-domain Sinkhorn for stability
    log_K = -C / epsilon  # [B, N, M]
    log_u = torch.zeros((B, N), device=x.device, dtype=x.dtype)
    log_v = torch.zeros((B, M), device=x.device, dtype=x.dtype)

    for _ in range(n_iters):
        # log_u = log_a - logsumexp(log_K + log_v, dim=2)
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
        # log_v = log_b - logsumexp(log_K^T + log_u, dim=2)
        log_v = log_b - torch.logsumexp(log_K.transpose(1, 2) + log_u.unsqueeze(1), dim=2)

    # Transport plan P = diag(u) K diag(v) in log space
    log_P = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)  # [B, N, M]
    P = torch.exp(log_P)

    # Sinkhorn cost: <P, C>
    cost = torch.sum(P * C, dim=(1, 2))  # [B]
    return cost


def wasserstein_loss_complex(y_pred, y_gt, epsilon=1, n_iters=10):
    """
    y_pred: [B, n_pts] complex tensor
    y_gt:   [B, n_pts] complex tensor
    returns: [B] W-loss
    """
    assert torch.is_complex(y_pred) and torch.is_complex(y_gt)
    # Treat each complex value as a 2D point (real, imag): point cloud size = n_pts
    x = torch.stack([y_pred.real, y_pred.imag], dim=-1)  # [B, n_pts, 2]
    y = torch.stack([y_gt.real, y_gt.imag], dim=-1)      # [B, n_pts, 2]
    return sinkhorn_wasserstein_cost(x, y, epsilon=epsilon, n_iters=n_iters, p=2)


def symmetric_complex_wloss(y_pred, y_gt, sym_loss_factor=2, epsilon=0.1, n_iters=50):
    """
    y_pred: [sym_loss_factor * batch_size, n_pts] (complex)
    y_gt:   [sym_loss_factor * batch_size, n_pts] (complex)

    returns:
      min_distances:  [batch_size]   (per-sample min over sym paths)
      activated_paths:[batch_size]   (argmin index in [0, sym_loss_factor-1])
    """
    n_pts = y_gt.shape[-1]
    y_pred = y_pred.reshape(sym_loss_factor, -1, n_pts)
    y_gt = y_gt.reshape(sym_loss_factor, -1, n_pts)

    sym, B, _ = y_pred.shape

    # Flatten sym into batch to compute W-loss in one go
    yp = y_pred.reshape(sym * B, n_pts)
    yg = y_gt.reshape(sym * B, n_pts)

    w_cost = wasserstein_loss_complex(yp, yg, epsilon=epsilon, n_iters=n_iters)  # [sym*B]
    w_cost = w_cost.reshape(sym, B)  # [sym, B]

    min_distances, activated_paths = torch.min(w_cost, dim=0)
    return min_distances, activated_paths

def symmetric_complex_loss(y_pred, y_gt, sym_loss_factor=2):
    """
    y_pred: [sym_loss_factor * batch_size, n_pts]
    y_gt: [sym_loss_factor * batch_size, n_pts]
    sym_loss_factor: int
    """
    n_pts = y_gt.shape[-1]
    y_pred = y_pred.reshape(sym_loss_factor, -1, n_pts)
    y_gt = y_gt.reshape(sym_loss_factor, -1, n_pts)



    distance_double = torch.mean(((y_pred.real - y_gt.real) ** 2+(y_pred.imag - y_gt.imag) ** 2), -1)
    #distance_double = torch.mean((torch.abs(y_pred.real - y_gt.real) +torch.abs(y_pred.imag - y_gt.imag) ), -1)
    min_distances, activated_paths = torch.min(distance_double, 0)
    #mean_distance=torch.mean(distance_double, 0)
    #return min_distances
    return min_distances, activated_paths

def symmetric_complex_softloss(y_pred, y_gt, sym_loss_factor=2, temp=0.1):
    """Marginalized (soft-min) version of symmetric_complex_loss.

    Instead of hard best-of-N (min over the sym in-plane hypotheses), take a
    softmax-weighted average of the per-hypothesis distances:
        w_k = softmax(-d_k / temp_eff),   loss = sum_k w_k d_k
    with temp_eff = temp * mean(d) so the temperature is scale-free w.r.t. the
    loss magnitude. temp->0 recovers the hard min; larger temp spreads gradient
    to all plausible poses (a probabilistic amortized pose). Still one encoder
    pass per particle -- no SO(3) search. Returns (loss[B], argmin_path[B]).
    """
    n_pts = y_gt.shape[-1]
    y_pred = y_pred.reshape(sym_loss_factor, -1, n_pts)
    y_gt = y_gt.reshape(sym_loss_factor, -1, n_pts)
    distance_double = torch.mean(((y_pred.real - y_gt.real) ** 2 + (y_pred.imag - y_gt.imag) ** 2), -1)
    temp_eff = temp * (distance_double.detach().mean() + 1e-8)
    weights = torch.softmax(-distance_double / temp_eff, dim=0)
    soft_distances = torch.sum(weights * distance_double, dim=0)
    _, activated_paths = torch.min(distance_double, 0)
    return soft_distances, activated_paths

def symmetric_complex_L1_loss(y_pred, y_gt, sym_loss_factor=2):
    n_pts = y_gt.shape[-1]
    y_pred = y_pred.reshape(sym_loss_factor, -1, n_pts)
    y_gt = y_gt.reshape(sym_loss_factor, -1, n_pts)
    #distance_double = torch.mean(y_pred - y_gt, -1)
    distance_double = torch.mean((torch.abs(y_pred.real - y_gt.real) +torch.abs(y_pred.imag - y_gt.imag) ), -1)

    min_distances, activated_paths = torch.min(distance_double, 0)
    #return min_distances.mean()
    return min_distances, activated_paths

def symmetric_loss(y_pred, y_gt, sym_loss_factor=2):
    """
    y_pred: [sym_loss_factor * batch_size, n_pts]
    y_gt: [sym_loss_factor * batch_size, n_pts]
    sym_loss_factor: int
    """
    n_pts = y_gt.shape[-1]
    y_pred = y_pred.reshape(sym_loss_factor, -1, n_pts)
    y_gt = y_gt.reshape(sym_loss_factor, -1, n_pts)
    distance_double = torch.mean((y_pred - y_gt) ** 2, -1)

    min_distances, activated_paths = torch.min(distance_double, 0)
    return min_distances.mean()



def Complex_L1_Loss(comp1,comp2):
  return (torch.log(real(comp1)-real(comp2)) + torch.log( imag(comp1)-imag(comp2))).mean()


def Complex_L2_Loss(comp1,comp2):
  return torch.log(( (real(comp1)-real(comp2)) **2 + (imag(comp1)-imag(comp2) )**2 ).mean())

def Complex_new_L2(comp1,comp2):
  return (torch.mean( (real(comp1)-real(comp2))**2 ,1)+ torch.mean( (imag(comp1)-imag(comp2))**2 ,1) ).mean()

def Complex_L2(comp1,comp2):
  return torch.sum( (real(comp1)-real(comp2)) **2 + (imag(comp1)-imag(comp2) )**2 ,1).mean()

def Complex_log_L2(comp1,comp2):
  return torch.sum( (real(comp1)-real(comp2)) **2 + (imag(comp1)-imag(comp2) )**2 ,1).mean()

def real(tensor):
    if not torch.is_complex(tensor):
        return tensor
    else:
        return tensor.real

def imag(tensor):
    if not torch.is_complex(tensor):
        return 0.
    else:
        return tensor.imag


def gen_coords_grid(imgsize,grid_type='2d',shift=True,scale=1.):
  xgrid = np.linspace(-1*scale, 1*scale, imgsize)
  ygrid = np.linspace(-1*scale, 1*scale, imgsize)
  assert grid_type=='2d' or grid_type=='3d', "Unknown grid type for "+str(grid_type)
  if grid_type=='2d':
    zgrid=np.zeros(imgsize*imgsize)
    x,y=np.meshgrid(xgrid, ygrid)
    coords = np.stack([x.ravel(), y.ravel(),zgrid.ravel()], 1)
    x_coord = torch.from_numpy(coords).float()
    if shift:
      x_coord = shift_coords(x_coord, scale, scale, 0., imgsize, imgsize, 1)

  elif grid_type=='3d':
    zgrid = np.linspace(-1*scale, 1*scale, imgsize)
    x,y,z= np.meshgrid(xgrid, ygrid,zgrid)
    coords = np.stack([x.ravel(), y.ravel(),z.ravel()], 1)
    x_coord = torch.from_numpy(coords).float()
    if shift:
      x_coord = shift_coords(x_coord, scale, scale, scale, imgsize, imgsize, imgsize)

  return x_coord



def gen_coords_grid_2d(imgsize,scale=1):
  xgrid = np.linspace(-1*scale, 1*scale, imgsize)
  ygrid = np.linspace(-1*scale, 1*scale, imgsize)


  x,y=np.meshgrid(xgrid, ygrid)
  coords = np.stack([x.ravel(), y.ravel()], 1)
  x_coord = torch.from_numpy(coords).float()


  return x_coord


def reverse_coords(x,imgsize):
  return (x+1)*(imgsize-1)/2

# def s2s2_to_matrix(v1, v2=None):
#     """
#     Normalize 2 3-vectors. Project second to orthogonal component.
#     Take cross product for third. Stack to form SO matrix.
#     """
#     if v2 is None:
#         assert v1.shape[-1] == 6
#         v2 = v1[..., 3:]
#         v1 = v1[..., 0:3]
#     u1 = v1
#     e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
#     u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
#     e2 = u2 / u2.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
#     e3 = torch.cross(e1, e2)
#     return torch.cat([e1[..., None, :], e2[..., None, :], e3[..., None, :]], -2)


def af2_quater_matrix(theta):
    b=theta.size(0)
    qb= theta[:,0]
    qc= theta[:,1]
    qd= theta[:,2]

    qa=Variable(theta.data.new_ones([b]))


    qnorm=torch.sqrt(qa**2+qb**2+qc**2+qd**2)

    qa=qa/qnorm
    qb=qb/qnorm
    qc=qc/qnorm
    qd=qd/qnorm

    rot_matrix = Variable(qa.data.new(b,3,3).zero_()).float()

    rot_matrix[:,0,0]= qa**2+qb**2-qc**2-qd**2
    rot_matrix[:,0,1]= 2*qb*qc-2*qa*qd
    rot_matrix[:,0,2]= 2*qb*qd+2*qa*qc
    rot_matrix[:,1,0]= 2*qb*qc+2*qa*qd
    rot_matrix[:,1,1]= qa**2-qb**2+qc**2-qd**2
    rot_matrix[:,1,2]= 2*qc*qd-2*qa*qb
    rot_matrix[:,2,0]= 2*qb*qd-2*qa*qc
    rot_matrix[:,2,1]= 2*qc*qd+2*qa*qb
    rot_matrix[:,2,2]= qa**2-qb**2-qc**2+qd**2

    return rot_matrix

def compute_rotational_average(image):
    # Assume image.shape is (N, N) and N is odd
    N = image.shape[0]

    # Create a grid of (x, y) coordinates
    x = torch.arange(N, dtype=torch.float32)
    y = torch.arange(N, dtype=torch.float32)
    x, y = torch.meshgrid(x, y)

    # Compute the radial coordinate for every grid point.
    # Grid points are arranged in a circle around the center of the image.
    center = (N - 1) // 2
    r = torch.sqrt((x - center)**2 + (y - center)**2)

    v=torch.zeros(center+1)
    masks=[]
    for i in range(center+1):
      masks.append(r<=i)


    # print(masks)

    for i in range(center):
      current_mask= torch.logical_and( torch.logical_not(masks[i]),masks[i+1])
      v[i+1]=torch.mean(image[current_mask])
    return v


def compute_rotational_std(image):
    # Assume image.shape is (N, N)
    N = image.shape[-1]

    # Create a grid of (x, y) coordinates
    x = torch.arange(N, dtype=torch.float32)
    y = torch.arange(N, dtype=torch.float32)
    x, y = torch.meshgrid(x, y)

    # Compute the radial coordinate for every grid point.
    # Grid points are arranged in a circle around the center of the image.
    center = N // 2
    r = torch.sqrt((x - center)**2 + (y - center)**2)

    # Flatten r values and sort them, keep track of indices for sorting
    sort_indices = torch.argsort(r.view(-1))
    r_sorted = r.view(-1)[sort_indices]  # radial coordinates sorted
    i_sorted = image.view(-1)[sort_indices]  # image intensities sorted

    # Get the indices of unique entries in the sorted radial coordinate array,
    # these are the indices at which to split the intensity array for averaging.
    r_uniq = torch.unique(r_sorted, return_inverse=True)[1]

    # Compute the mean for each radial distance
    bin_counts = torch.bincount(r_uniq)
    # mean = torch.bincount(r_uniq, weights=i_sorted) / bin_counts

    # # Compute the sum of squared differences
    # sum_sq_diff = torch.bincount(r_uniq, weights=(i_sorted - mean[r_uniq])**2)

    # # Compute the standard deviation
    # std = torch.sqrt(sum_sq_diff / bin_counts)




    # Compute the mean for each radial distance
    real_mean = torch.bincount(r_uniq, weights=i_sorted.real )/ bin_counts
    imag_mean = torch.bincount(r_uniq, weights=i_sorted.imag )/ bin_counts

    # Compute the sum of squared differences
    real_sum_sq_diff = torch.bincount(r_uniq, weights=(i_sorted.real - real_mean[r_uniq])**2)
    imag_sum_sq_diff = torch.bincount(r_uniq, weights=(i_sorted.imag - imag_mean[r_uniq])**2)

    # Compute the standard deviation
    real_std = torch.sqrt(real_sum_sq_diff / bin_counts)
    imag_std = torch.sqrt(imag_sum_sq_diff / bin_counts)

    # Combine the real and imaginary parts
    std = torch.sqrt(real_std**2 + imag_std**2)

    # Return the result up to imgsize//2 (Nyquist limit)
    return std[:N // 2]


def create_image_from_radial_profile(profile):
    # Determine the size of the image to be created
    N = profile.shape[0] * 2

    # Create a grid of (x, y) coordinates
    x = torch.arange(N, dtype=torch.float32)
    y = torch.arange(N, dtype=torch.float32)
    x, y = torch.meshgrid(x, y)

    # Compute the radial coordinate for every grid point.
    center = N // 2
    r = torch.sqrt((x - center)**2 + (y - center)**2)

    # Round and clamp the radial coordinates to use them as indices
    r = r.round().long()
    r = torch.clamp(r, 0, profile.shape[0]-1)

    # Map the radial coordinates to their corresponding values in the profile
    image = profile[r]

    return image

def generate_quater(theta):
  b=theta.size(0)
  qb= theta[:,0]
  qc= theta[:,1]
  qd= theta[:,2]
  tensor_one = torch.ones(b).to(qb.device)
  tensor_one=tensor_one*1e-8

  float_q=torch.max(tensor_one,qb**2+qc**2+qd**2)
  qnorm=torch.sqrt(float_q)
  vx=qb/qnorm
  vy=qc/qnorm
  vz=qd/qnorm

  sin_rot=theta[:,3]
  comp=torch.max(tensor_one,1-sin_rot**2)
  cos_rot=torch.sqrt(comp)

  qa=cos_rot
  qb=vx*sin_rot
  qc=vy*sin_rot
  qd=vz*sin_rot

  quater_input=torch.stack((qa,qb,qc,qd),dim=1)
  return quater_input

def gen_mask(imgsize,radius,grid_type='2d',flip=False,shift=None,soft_edge=0.0):
  '''Spherical/circular mask.

  soft_edge (fraction of imgsize): if >0, the mask falls off from 1 to 0 with a
  raised-cosine (Hann) taper of that width just outside `radius`, instead of a hard
  binary cut. A hard cut in Fourier space produces sinc ringing (Gibbs) in real space
  (and vice-versa); a soft edge suppresses it. soft_edge=0 reproduces the original
  hard binary mask exactly, so existing call sites are unchanged.
  '''
  radius = imgsize*radius
  assert grid_type=='2d' or grid_type=='3d', "Unknown grid type for "+str(grid_type)
  # update centerid method.
  #center_id= (imgsize+1.)/2
  # this is exactly the perf
  center_id=imgsize//2
  if grid_type=='2d':
    y_grid, x_grid = np.ogrid[:imgsize,:imgsize]

    center = np.array([center_id, center_id])
    if shift is not None:
        center[0]+=shift[0]
        center[1]+=shift[1]
    dist = np.sqrt((center[0] - y_grid)**2 + (center[1] - x_grid)**2)
  else:
    xg,yg,zg=np.ogrid[:imgsize,:imgsize,:imgsize]
    center=np.array([center_id, center_id,center_id])
    if shift is not None:
        center[0]+=shift[0]
        center[1]+=shift[1]
        center[2]+=shift[2]

    dist = np.sqrt((center[0] - xg)**2 + (center[1] - yg)**2 + (center[2]-zg)**2)

  if soft_edge and soft_edge > 0:
    w = max(imgsize*soft_edge, 1e-6)
    # 1 inside `radius`, raised-cosine down to 0 across the next `w` pixels.
    mask = 0.5*(1.0+np.cos(np.pi*np.clip((dist-radius)/w, 0.0, 1.0)))
    if flip:
      mask = 1.0 - mask
    mask = torch.from_numpy(mask).float()
  else:
    if flip:
      mask = (torch.from_numpy(dist) >= radius).float()
    # elif single:
    #   mask = torch.from_numpy(dist) < radius
    else:
      mask = (torch.from_numpy(dist) <= radius).float()

  if grid_type=='2d':
    mask = mask.view(imgsize,imgsize)
  else:
    mask = mask.view(imgsize,imgsize,imgsize)
  return mask

def gen_shiftmasks_torch(imgsize,radius,grid_type='2d',flip=False,shift=None,device=None):
  radius = imgsize*radius
  assert grid_type=='2d' or grid_type=='3d', "Unknown grid type for "+str(grid_type)
  # shift shape : [b,2]
  b=shift.size(0)
  center_id=imgsize/2
  if grid_type=='2d':
    x_grid=torch.arange(0,imgsize,device=device).view(1,1,imgsize).expand(b,1,imgsize)
    y_grid=torch.arange(0,imgsize,device=device).view(1,imgsize,1).expand(b,imgsize,1)

    center = torch.tensor([center_id, center_id],dtype=torch.long,device=device).unsqueeze(0).expand(b,2)

    if shift is not None:
      center=center+shift

    dist =torch.sqrt(    (center[:,0].unsqueeze(-1).unsqueeze(-1)  - y_grid )**2 +  (center[:,1].unsqueeze(-1).unsqueeze(-1)  - x_grid )**2    )
    if flip:
      masks=dist >= radius
    else:
      masks=dist <= radius

    masks = masks.view(b,imgsize,imgsize).float()

  return masks


def which_half_space(coords, eps=1e-6):
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    slab_xyz = (x < -eps)
    slab_yz = torch.logical_and(torch.logical_and(x > -eps, x < eps), y < -eps)
    slab_z = torch.logical_and(torch.logical_and(torch.logical_and(x > -eps, x < eps),
                                                 torch.logical_and(y > -eps, y < eps)), z < -eps)

    return torch.logical_or(slab_xyz, torch.logical_or(slab_yz, slab_z))

def where_DC(coords, eps=1e-6):
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    slab_x = torch.logical_and(x > -eps, x < eps)
    slab_y = torch.logical_and(y > -eps, y < eps)
    slab_z = torch.logical_and(z > -eps, z < eps)

    return torch.logical_and(slab_x, torch.logical_and(slab_y, slab_z))


class Symmetrizer():
    def __init__(self):
        """
        Initialization of a Symmetrizer, to enforce symmetry in Fourier space.
        """
        self.half_space_indicator = None
        self.DC_indicator = None

    def initialize(self, coords):
        self.half_space_indicator = which_half_space(coords)
        self.DC_indicator = where_DC(coords)

    def symmetrize_input(self, coords):
        # Place the "negative" coords in the "positive" half space
        coords[self.half_space_indicator] = -coords[self.half_space_indicator]
        return coords
    def antisymmetrize_output(self, output):
        # Flip the imaginary part on the "negative" half space and force DC component to be zero
        # print(output.size())
        # print('')
        batch_sz = output.shape[0]
        N = output.shape[1]
        #N=1
        output = output.reshape(batch_sz, N, -1, 2)
        # output.shape = Batch, N, channels, 2
        channels = output.shape[2]
        half_space = self.half_space_indicator.reshape(batch_sz, N, 1, 1).repeat(1, 1, channels, 2)
        DC = self.DC_indicator.reshape(batch_sz, N, 1, 1).repeat(1, 1, channels, 2)
        output_sym = torch.where(half_space, torch.cat((output[..., 0].unsqueeze(-1),
                                                        -output[..., 1].unsqueeze(-1)), dim=-1), output)
        output_sym_DC = torch.where(DC, torch.cat((output_sym[..., 0].unsqueeze(-1),
                                                torch.zeros_like(output_sym[..., 0].unsqueeze(-1))), dim=-1),
                                 output_sym)
        output_sym_DC = output_sym_DC.reshape(batch_sz, N, -1)
        return output_sym_DC

# def compute_ctf(nx, ny,  dfu, dfv, defocus_angle, pixel_size=2.2,volt=300, cs=2.7, w=0.07,phase_shift=0, b_factor=None):
#     """
#     Compute the 2D CTF

#     Input:
#         freqs (np.ndarray) Nx2 or BxNx2 tensor of 2D spatial frequencies
#         dfu (float or Bx1 tensor): DefocusU (Angstrom)
#         dfv (float or Bx1 tensor): DefocusV (Angstrom)
#         defocus_angle (float or Bx1 tensor): DefocusAngle (degrees)
#         volt (float or Bx1 tensor): accelerating voltage (kV)
#         cs (float or Bx1 tensor): spherical aberration (mm)
#         w (float or Bx1 tensor): amplitude contrast ratio
#         phase_shift (float or Bx1 tensor): degrees
#         b_factor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
#     """
#     fx = torch.fft.fftfreq(nx, d=pixel_size).to(dfu.device)
#     fy = torch.fft.fftfreq(ny, d=pixel_size).to(dfu.device)
#     x, y = torch.meshgrid(fx, fy).to(dfu.device)

#     # convert units
#     volt = volt * 1000
#     cs = cs * 10 ** 7
#     #defocus_angle = defocus_angle * np.pi / 180
#     phase_shift = phase_shift * np.pi / 180
#     b=dfv.size(0)
#     # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
#     lam = 12.2639 / (volt + 0.97845e-6 * volt ** 2) ** .5
#     # x = freqs[..., 0]
#     # y = freqs[..., 1]
def torch_load(filepath,dtype=np.int32):
  return torch.from_numpy(np.array(DL.read_mrc(filepath),dtype=dtype))


def list2tensor(l):
  return torch.from_numpy(np.array(l,dtype=np.float32))

def get_ctf(ctf_params_dict,nx,ny):
  dfu=list2tensor(ctf_params_dict['dfu'])
  dfv=list2tensor(ctf_params_dict['dfv'])
  dfang=np.pi*list2tensor(ctf_params_dict['dfang'])/180.

  #print (dfu.size(),dfv.size(),dfang.size())
  volt=ctf_params_dict['volt']
  cs=ctf_params_dict['cs']
  ac=ctf_params_dict['ampcont']
  pixel_size=ctf_params_dict['apix']
  full_ctf=create_ctf(nx,ny,dfu, dfv, dfang, volt, cs, ac, pixel_size)
  return dt(full_ctf)



def cuda_get_ctf(ctf_params_dict,nx,ny,downsize=1):
    device=torch.device('cuda')

    dfu=ctf_params_dict['dfu']
    dfv=ctf_params_dict['dfv']
    #dfang=np.pi*list2tensor(ctf_params_dict['dfang'])/180
    dfang=ctf_params_dict['dfang']
    b=len(dfu)
    volt=ctf_params_dict['volt']
    cs=ctf_params_dict['cs']
    ac=ctf_params_dict['ampcont']
    pixel_size=ctf_params_dict['apix']

    cuda_block=8192
    imgsize=nx
    batch=max(1,int(b/cuda_block))
    imgsize=int(imgsize/downsize)
    ctfs=torch.zeros(b,imgsize,imgsize)

    if batch*cuda_block<b:
      batch+=1
    finished=0
    for i in range(batch):
      batch_dfu=list2tensor(dfu[i*cuda_block:(i+1)*cuda_block]).detach().to(device)
      batch_dfv=list2tensor(dfv[i*cuda_block:(i+1)*cuda_block]).detach().to(device)
      batch_dfang=(np.pi*list2tensor(dfang[i*cuda_block:(i+1)*cuda_block])/180).detach().to(device)
      if downsize==1:
        ctfs[i*cuda_block:(i+1)*cuda_block]=create_ctf(imgsize,imgsize,batch_dfu,batch_dfv,batch_dfang, volt, cs, ac, pixel_size).view(batch_dfu.size(0),imgsize,imgsize).detach()
      #else:
      #  ctfs[i*cuda_block:(i+1)*cuda_block]=torch.nn.AvgPool2d(downsize)(create_ctf(nx,nx,batch_dfu,batch_dfv,batch_dfang, volt, cs, ac, pixel_size).view(batch_dfu.size(0),1,nx,nx)).view(batch_dfu.size(0),imgsize,imgsize).detach()
      torch.cuda.empty_cache()
      finished+=batch_dfu.size(0)
      print(str(finished)+'/'+str(b),"\r",end="",flush=True)
    print('\n')
    print('Done Calculating CTF.')
    return ctfs


def create_ctf(nx, ny, dfu, dfv, dfang, volt=300, cs=2.7, ac=0.07, pixel_size=2.2,b_factor=None):
    # Calculate the Fourier space coordinates
    fx = torch.fft.fftfreq(nx, d=pixel_size)
    fy = torch.fft.fftfreq(ny, d=pixel_size)
    x, y = torch.meshgrid(fx, fy)
    try:
      b=dfv.size(0)
    except:
      #print(dfv)
      b=1

    volt = volt * 1000    # Convert kilovots to volts
    cs = cs * 10**7       # Convert spherical aberation from mm to A

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2

    lam = 12.2639 / np.sqrt(volt + 0.97845e-6 * volt**2)
    #lam = 12.2643247 / torch.sqrt(volt + volt**2 * 0.978466e-6)  # from cryoSPARC

    # x = freqs[:,0]
    # y = freqs[:,1]
    #ang=torch.atan2(y,x)

    ang = torch.atan2(y,x).expand(b,nx,ny).to(dfu.device)
    #print(ang.device)
    dfu=dfu.unsqueeze(-1).unsqueeze(-1)
    dfv=dfv.unsqueeze(-1).unsqueeze(-1)
    dfang=dfang.unsqueeze(-1).unsqueeze(-1)
    s2 = (x**2 + y**2).to(dfu.device)
    df = 0.5*(dfu + dfv + (dfu-dfv)*torch.cos(2*(ang-dfang)))

    gamma = 2*torch.pi*(-0.5*df*lam*s2 + 0.25*cs*lam**3*s2**2)
    ctf = np.sqrt(1-ac**2)*torch.sin(gamma) - ac*torch.cos(gamma)
    if b_factor is not None:
        ctf *= torch.exp(-b_factor / 4 * s2)
    return -torch.fft.fftshift(ctf,dim=(-2,-1))
    #return ctf
    #return -torch.fft.fftshift(ctf,dim=(-2,-1))

def b2a(old):
  phase_shift= np.exp(-2j * np.pi * (np.arange(old.shape[0])[:,None] +  np.arange(old.shape[1])[None,:])/2 )
  return old*phase_shift

def dt(t):
    return t.detach().cpu().numpy()





 # def translate_ft(input_fourier_image, shift):
 #        '''
 #        Translate an image by phase shifting its Fourier transform

 #        Inputs:
 #            img: FT of image (B x img_dims x 2)
 #            t: shift in pixels (B x T x 2)
 #            mask: Mask for lattice coords (img_dims x 1)

 #        Returns:
 #            Shifted images (B x T x img_dims x 2)

 #        img_dims can either be 2D or 1D (unraveled image)
 #        '''
 #        # F'(k) = exp(-2*pi*k*x0)*F(k)
 #        coords = self.freqs2d if mask is None else self.freqs2d[mask]
 #        img = img.unsqueeze(1) # Bx1xNx2
 #        t = t.unsqueeze(-1) # BxTx2x1 to be able to do bmm
 #        tfilt = coords @ t * -2 * np.pi # BxTxNx1
 #        tfilt = tfilt.squeeze(-1) # BxTxN
 #        c = torch.cos(tfilt) # BxTxN
 #        s = torch.sin(tfilt) # BxTxN
 #        return torch.stack([img[...,0]*c-img[...,1]*s,img[...,0]*s+img[...,1]*c],-1)

# def phase_shift_ft(input_fourier_image, shift):
#     # if input_fourier_image.data.type()!='torch.FloatTensor':
#     #     input_fourier_image=input_fourier_image.float()
#     device=input_fourier_image.device
#     shift=shift.to(device)
#     assert input_fourier_image.size(0)==shift.size(0)
#     b,img_size,img_size = input_fourier_image.size()
#     # Added at Jul 11, 2023. change origin to up-left corner
#     dx,dy=-1*shift[:,0],shift[:,1]
#     dx=dx.unsqueeze(-1).unsqueeze(-1)
#     dy=dy.unsqueeze(-1).unsqueeze(-1)
#     shift_x = dx / img_size
#     shift_y = dy / img_size
#     x,y=torch.meshgrid(torch.arange(-img_size/2,img_size/2),torch.arange(-img_size/2,img_size/2))
#     x=x.expand(b,img_size,img_size).to(device)
#     y=y.expand(b,img_size,img_size).to(device)

#     phasor_shift = torch.exp(-2j * torch.pi * (shift_x *x  + shift_y * y))

    

#     output_fourier_image = input_fourier_image * phasor_shift
#     return output_fourier_image


# def ift2(t):
#     if t.data.type()!='torch.FloatTensor':
#         t=t.float()
#     return torch.fft.ifftn(torch.fft.ifftshift(t,dim=(-2,-1)),dim=(-2,-1)).real

def ift2(t,norm="backward"):
    # if t.data.type()!='torch.FloatTensor':
    #     t=t.float()
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(t,dim=(-2,-1)),dim=(-2,-1),norm=norm),dim=(-2,-1)).real

def ift3(t):
    # if t.data.type()!='torch.FloatTensor':
    #     t=t.float()
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(t,dim=(-3,-2,-1)),dim=(-3,-2,-1)),dim=(-3,-2,-1)).real


def ft2(t,norm="backward"):
    # use norm instead to avoid inf when using  float16
    if t.data.type()!='torch.FloatTensor':
        t=t.float()
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(t,dim=(-2,-1)),dim=(-2,-1),norm=norm),dim=(-2,-1))


def ft3(t):
    # if t.data.type()!='torch.FloatTensor':
    #     t=t.float()
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(t,dim=(-3,-2,-1)),dim=(-3,-2,-1)),dim=(-3,-2,-1))

def shift_coords(coords, x_range, y_range, z_range, Nx, Ny, Nz, flip=False):
    """
    Shifts the coordinates and puts the DC component at (0, 0, 0).

    Parameters
    ----------
    coords: torch.tensor (..., 3)
    x_range: float
        (max_x - min_x) / 2
    y_range: float
        (max_y - min_y) / 2
    z_range: float
        (max_z - min_z) / 2
    Nx: int
    Ny: int
    Nz: int
    flip: bool

    Returns
    -------
    coords: torch.tensor (..., 3)
    """
    alpha = -1.
    if flip:  # "unshift" the coordinates.
        alpha = 1.


    if Nx % 2 == 0:
        x_shift = coords[..., 0] + alpha * x_range / (Nx - 1)
    else:
        x_shift = coords[..., 0]
    if Ny % 2 == 0:
        y_shift = coords[..., 1] + alpha * y_range / (Ny - 1)
    else:
        y_shift = coords[..., 1]
    if Nz % 2 == 0:
        z_shift = coords[..., 2] + alpha * z_range / (Nz - 1)
    else:
        z_shift = coords[..., 2]
    coords = torch.cat((x_shift.unsqueeze(-1),
                        y_shift.unsqueeze(-1),
                        z_shift.unsqueeze(-1)), dim=-1)
    return coords




def add_gaussian_noise_fourier_space(T, SNR):
    # Calculate the power of the signal
    signal_power = torch.sum(torch.abs(T)**2) / T.numel()

    # Calculate the power of the noise according to the desired SNR
    noise_power = signal_power / SNR

    # Generate Gaussian noise with the desired noise power
    noise=torch.randn_like(T) * torch.sqrt (noise_power)

    # Add the noise to the original Fourier space image
    T_noisy = T + noise

    return T_noisy

