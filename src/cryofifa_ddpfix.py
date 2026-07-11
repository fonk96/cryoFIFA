from __future__ import print_function, division

import os

_LOCAL_RANK_EARLY = os.environ.get("LOCAL_RANK")
if _LOCAL_RANK_EARLY is not None:
  import torch
  torch.cuda.set_device(int(_LOCAL_RANK_EARLY))

import warnings
warnings.filterwarnings('ignore')


from new_dataloader import StarDataSet

import numpy as np
import math
import contextlib
import torch.utils.checkpoint
import pandas as pd
import sys
from cam_module import *
import gc
from new_dataloader import LmdbDataSet

from scipy.spatial.transform import Rotation 
import lmdb
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import torchvision

import json
import torch.distributed as distr

import os
import sys
from merged_models import *

from utils import *
from new_utils import *

import torch.optim as optim

from fairscale.optim.oss import OSS

from torch.nn.parallel import  DistributedDataParallel as DDP
from torch.nn.parallel import  DataParallel

from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing as mp

import time

import datetime
from fairscale.optim.grad_scaler import ShardedGradScaler

from torch.profiler import profile

np.set_printoptions(precision=4)


def _bare(m):
  # unwrap DDP so we save/load the underlying module's weights
  return m.module if hasattr(m, 'module') else m


def enc_fwd(model, x, args):
  """Encoder forward with optional Tier-2 channels_last input + activation checkpointing.
  channels_last: NHWC input so V100 conv tensor cores are faster (the model is also converted
  to channels_last at construction). grad_checkpoint: recompute activations in backward to save
  memory (only when training -- no benefit under eval/no-grad)."""
  if args.channels_last:
    x = x.contiguous(memory_format=torch.channels_last)
  if args.grad_checkpoint and model.training:
    return torch.utils.checkpoint.checkpoint(model, x, use_reentrant=False)
  return model(x)


def dec_fwd(model, coords, latent, args):
  """Decoder (SIREN) forward with optional activation checkpointing. The decoder is the memory
  hog in the refine phase (it decodes ~pi*r^2 coords x sym at the large frontier and stores all
  its mul/addmm activations); checkpointing it recomputes them in backward to save memory."""
  if args.grad_checkpoint and model.training:
    if latent is None:
      return torch.utils.checkpoint.checkpoint(model, coords, use_reentrant=False)
    return torch.utils.checkpoint.checkpoint(model, coords, latent, use_reentrant=False)
  return model(coords) if latent is None else model(coords, latent)


class PoseRefiner(nn.Module):
  """Option-B conditioned-cascade pose refiner. Takes the render-and-compare RESIDUAL image
  [b,1,H,W] = ift2(y - pred@coarse_pose) and outputs a SMALL pose update (dw in so(3) tangent,
  dt in pixels). The final Linear is zero-initialised so it starts as the identity map (no
  correction) -> a natural warm start; it then learns the amortised Newton step
  (data-model residual -> pose update)."""
  def __init__(self, in_ch=1):
    super(PoseRefiner, self).__init__()
    self.net = nn.Sequential(
        nn.Conv2d(in_ch, 16, 5, 2, 2), nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, 5, 2, 2),    nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 5, 2, 2),    nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1), nn.Flatten())
    self.head = nn.Linear(64, 5)
    nn.init.zeros_(self.head.weight)
    nn.init.zeros_(self.head.bias)
  def forward(self, x):
    return self.head(self.net(x))


def lr_at_epoch(epoch, base_lr, min_lr, total_epochs, warmup_epochs, use_warmup, schedule):
  """Per-epoch LR. cosine decays base_lr->min_lr over [warmup, total], so the minimum is only
  reached at the end -- high-res shells that open late in frequency-marching are still learned.
  schedule='none' returns base_lr (original fixed-LR behavior)."""
  if schedule == 'none':
    return base_lr
  if use_warmup and warmup_epochs > 0 and epoch < warmup_epochs:
    return min_lr + (base_lr - min_lr) * float(epoch + 1) / float(warmup_epochs)
  if schedule == 'cosine':
    denom = max(1, total_epochs - (warmup_epochs if use_warmup else 0))
    progress = float(epoch - (warmup_epochs if use_warmup else 0)) / float(denom)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
  return base_lr


def save_checkpoint(path, epoch, models, optimizers, scaler, args):
  ckpt = {'epoch': int(epoch), 'args': vars(args)}
  ckpt['scaler'] = scaler.state_dict() if scaler is not None else None
  for name, m in models.items():
    ckpt['model_' + name] = _bare(m).state_dict()
  for name, o in optimizers.items():
    ckpt['optim_' + name] = o.state_dict()
  tmp = path + '.tmp'
  torch.save(ckpt, tmp)
  os.replace(tmp, path)


def load_checkpoint(path, models, optimizers, scaler, map_location):
  ckpt = torch.load(path, map_location=map_location)
  for name, m in models.items():
    key = 'model_' + name
    if key in ckpt:
      _bare(m).load_state_dict(ckpt[key])
  for name, o in optimizers.items():
    key = 'optim_' + name
    if key in ckpt:
      o.load_state_dict(ckpt[key])
  if scaler is not None and ckpt.get('scaler') is not None:
    scaler.load_state_dict(ckpt['scaler'])
  return int(ckpt.get('epoch', -1)) + 1  # resume at the next epoch





def get_global_min_free_mem_gb(device):
  """Query only the current rank's GPU, then all-reduce the global minimum.
  This avoids creating CUDA contexts on every GPU from every rank.
  """
  free_bytes, _ = torch.cuda.mem_get_info(device)
  t = torch.tensor([free_bytes / 1024**3], device=device, dtype=torch.float32)
  distr.all_reduce(t, op=distr.ReduceOp.MIN)
  return float(t.item())


def ddp_wrap(module, local_rank, args):
  return DDP(
    module,
    device_ids=[local_rank],
    output_device=local_rank,
    broadcast_buffers=False,
    gradient_as_bucket_view=args.fast_ddp,
  )

def main():
  args=get_parser()
  imgsize=args.imgsize
  mode=args.mode
  latent_code_dim_2d=args.latent_code_dim_2d
  latent_code_dim_3d=args.latent_code_dim_3d  
  mask_steps=args.mask_steps
  num_epochs=args.num_epochs
  render_number=args.render_number
  #===========setup_distributed==========#

  local_rank = int(os.environ["LOCAL_RANK"])
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  setup_for_distributed(rank == 0)

  torch.cuda.set_device(local_rank)
  device = torch.device(f"cuda:{local_rank}")

  distr.init_process_group(
    backend="nccl",
    init_method="env://",
  )

  # CPU-side synchronization group for long single-rank render/checkpoint sections.
  # This prevents other ranks from entering NCCL collectives while one rank is still writing maps.
  sync_pg = distr.new_group(
    backend="gloo",
    timeout=datetime.timedelta(hours=12),
  )

  #======================================#
  if not os.path.exists(args.render_path) and (local_rank==0):
    os.mkdir(args.render_path)
  start_time=time.time()

  #=================data loader======================#
   
  data_train,total_len,data_apix= data_constructor(args.data_path,args.particles_meta,imgsize,args.lmdb_path,with_name=args.save_pose,skip_pose=args.skip_read_pose)    
  # downsampled_data_train,total_len,donsampled_data_apix=data_constructor(args.data_path,args.particles_meta,imgsize,args.lmdb_path,downsample_train=args.downsample_train,downsample_imgsize=downsample_radius)

  sampler=DistributedSampler(data_train)
  # downsampled_sampler=DistributedSampler(downsampled_data_train)

  initial_batch_size=args.minibatch_size
  train_iterator_list=[]
  minibatch_slots=[]
  batch_slots=int(np.log(initial_batch_size)/(np.log(2)))
  for i in range(batch_slots):
    batchsize_slot=int(initial_batch_size/(2**i))
    minibatch_slots.append(batchsize_slot)
    #print(batchsize_slot)
    train_iterator_list.append (torch.utils.data.DataLoader(data_train, batch_size=batchsize_slot,shuffle=False,pin_memory=True,num_workers=args.num_workers,drop_last=False,sampler=sampler))
  current_dataloader_index=0
  
  
  #==========preprocessing module=========#
  pre_module=preprocesser(lowpass=True,gaussian=args.gaussian,preprocessing_output_channels=args.preprocessing_output_channels,imgsize=imgsize,kernel_size=11,kernel_variance=0.01,include_input=False)
  pre_module=pre_module.to(device)    

  downsampled_pre_module=preprocesser(lowpass=True,gaussian=args.gaussian,preprocessing_output_channels=args.preprocessing_output_channels,imgsize=2*args.downsample_radius,kernel_size=11,kernel_variance=0.01,include_input=False)

  downsampled_pre_module=downsampled_pre_module.to(device)    

  input_mask=gen_mask(imgsize,args.input_radius).view(imgsize,imgsize).to(device)


  _soft_edge = args.soft_mask_width if args.soft_mask else 0.0
  reconstruct_mask=gen_mask(imgsize,args.reconstruct_radius,soft_edge=_soft_edge).view(imgsize,imgsize).to(device)
  radwn=int(imgsize//2*data_apix/args.mask_lowpass)
  radwn=16
  lowpass_mask_3d=torch.from_numpy(gen_lowpassmask_3d(imgsize,radwn)).float().to(device)

  fourier_masks,new_fourier_masks=reconstruct_fourier_masks(start_resolution=args.start_resolution,data_apix=data_apix,imgsize=imgsize,downsample_train=False,mask_steps=args.mask_steps,butterworth=False,exp_mask=False,device=device,end_resolution=args.end_resolution)

  #noise_mask=gen_mask(imgsize,args.noise_radius,'2d',True).to(device)
  reconstruct_mask_3d=gen_mask(imgsize,args.reconstruct_radius,'3d',soft_edge=_soft_edge).to(device)

  #Tensor_croper=torchvision.transforms.CenterCrop(args.downsample_radius*2).to(device)
  #===============models==================#
  shift_dim=2
  fade_in_start_epoch=0  

  lr =args.learning_rate  
  base_optimizer_arguments = { "lr": lr,"eps": 1e-8}
  base_optimizer=torch.optim.Adam
  in_channels=args.preprocessing_output_channels  if args.preprocess else 1
  Encoder_Feature_Extractor = model_constructor(name='Feature_Extrator',backbone=args.encoder_backbone,basename=args.encoder_basename,in_channels=in_channels,image_size=imgsize,use_batchnorm=False)
 
  Encoder_Feature_Extractor.apply(weight_init)
  Encoder_Feature_Extractor=Encoder_Feature_Extractor.to(device)
  if args.channels_last:
    Encoder_Feature_Extractor=Encoder_Feature_Extractor.to(memory_format=torch.channels_last)
  #***********FAIRSCALE MODEL CONFIG*****************#   
  # Encoder_Feature_Extractor_Optimer=OSS(params=Encoder_Feature_Extractor.parameters(),optim=base_optimizer,**base_optimizer_arguments)
  # Encoder_Feature_Extractor=ShardedDDP(Encoder_Feature_Extractor,Encoder_Feature_Extractor_Optimer)

  Encoder_Feature_Extractor_Optimer=optim.Adam(Encoder_Feature_Extractor.parameters(),lr=lr)


  Encoder_Feature_Extractor = ddp_wrap(Encoder_Feature_Extractor, local_rank, args)
  if args.downsample_train:
    Encoder_Feature_Extractor_Local = model_constructor(name='Feature_Extrator',backbone=args.encoder_backbone,basename=args.encoder_basename,in_channels=in_channels,image_size=imgsize,use_batchnorm=False)
    Encoder_Feature_Extractor_Local=Encoder_Feature_Extractor_Local.to(device)
    if args.channels_last:
      Encoder_Feature_Extractor_Local=Encoder_Feature_Extractor_Local.to(memory_format=torch.channels_last)
    Encoder_Feature_Extractor_Optimer_Local=optim.Adam(Encoder_Feature_Extractor_Local.parameters(),lr=lr)  # full LR; warm-restart schedule + correction-magnitude scale handle local_refine_scale
    Encoder_Feature_Extractor_Local = ddp_wrap(Encoder_Feature_Extractor_Local, local_rank, args)
  if args.mode=='both':

    Encoder_Feature_Extractor_3d = model_constructor(name='Feature_Extrator',backbone=args.encoder_backbone,basename=args.encoder_basename,in_channels=args.preprocessing_output_channels+1,image_size=imgsize,use_batchnorm=False)

    Encoder_Feature_Extractor_3d=Encoder_Feature_Extractor_3d.to(device)
    Encoder_Feature_Extractor_Optimer_3d=optim.Adam(Encoder_Feature_Extractor_3d.parameters(),lr=lr)


    Encoder_Feature_Extractor_3d = ddp_wrap(Encoder_Feature_Extractor_3d, local_rank, args)
  if args.mode=='3d' or args.mode=='both':
    rotation_dim=6 

    Encoder_3D_Predictor = model_constructor(name='Encode_Predictor',rotation_dim=rotation_dim,shift_dim=2,latent_code_dim=args.latent_code_dim_3d,activation=args.encoder_activation,last_activation=args.last_activation,hidden_dim=args.encoder_hidden_dim,add_layers=args.encoder_add_layers)
    Encoder_3D_Predictor.apply(weight_init)
    Encoder_3D_Predictor=Encoder_3D_Predictor.to(device)  
    Encoder_3D_Predictor_Optimer=optim.Adam(Encoder_3D_Predictor.parameters(),lr=lr)

    Encoder_3D_Predictor = ddp_wrap(Encoder_3D_Predictor, local_rank, args)
    if args.downsample_train:
      Encoder_3D_Predictor_Local = model_constructor(name='Encode_Predictor',rotation_dim=rotation_dim,shift_dim=2,latent_code_dim=args.latent_code_dim_3d,activation=args.encoder_activation,last_activation=args.last_activation,hidden_dim=args.encoder_hidden_dim,add_layers=args.encoder_add_layers)
      Encoder_3D_Predictor_Local.apply(weight_init)
      Encoder_3D_Predictor_Local=Encoder_3D_Predictor_Local.to(device)
      # match the local feature-extractor LR (lr*local_refine_scale) so the whole local
      # refinement path is scaled consistently (was full lr -> inconsistent with efe_local).
      Encoder_3D_Predictor_Optimer_Local=optim.Adam(Encoder_3D_Predictor_Local.parameters(),lr=lr)  # full LR; see warm-restart schedule

      Encoder_3D_Predictor_Local = ddp_wrap(Encoder_3D_Predictor_Local, local_rank, args)
    Decoder_3D= model_constructor(name='Decoder',hidden_dim=args.decoder_hidden_dim, num_layers=args.decoder_layers,add_layers=args.decoder_add_layers,activation='SIREN',w0=args.w0,w=args.w,out_dim=2,in_dim=3,latent_code_dim=args.latent_code_dim_3d,use_symmetrizer=args.use_symmetrizer)
    Decoder_3D.apply(weight_init)
    Decoder_3D=Decoder_3D.to(device) 
    #***********FAIRSCALE MODEL CONFIG*****************#     
    # Decoder_3D_Optimer=OSS(params=Decoder_3D.parameters(),optim=base_optimizer,**base_optimizer_arguments)     
    # Decoder_3D=ShardedDDP(Decoder_3D,Decoder_3D_Optimer)
    Decoder_3D_Optimer=optim.Adam(Decoder_3D.parameters(),lr=lr)
    Decoder_3D = ddp_wrap(Decoder_3D, local_rank, args)
    if args.cascade_refine:
      Pose_Refiner = PoseRefiner().to(device)
      if args.channels_last:
        Pose_Refiner = Pose_Refiner.to(memory_format=torch.channels_last)
      Pose_Refiner_Optimer = optim.Adam(Pose_Refiner.parameters(), lr=lr)
      Pose_Refiner = ddp_wrap(Pose_Refiner, local_rank, args)
  #-------------constants----------------#
  # Constant grid used in phase_shift_ft function.  
  helper_x,helper_y=torch.meshgrid(torch.arange(-imgsize/2,imgsize/2),torch.arange(-imgsize/2,imgsize/2))
  helper_x=helper_x.to(device)
  helper_y=helper_y.to(device)   

  x_coord_2d=gen_coords_grid_2d(imgsize)
  x_coord_plane=gen_coords_grid(imgsize,'2d',args.using_shift_coords,args.x_scale)
  x_coord_3d=gen_coords_grid(imgsize,'3d',args.using_shift_coords,args.x_scale)


  #======================================#
  # scaler=ShardedGradScaler()

  scaler=torch.cuda.amp.GradScaler()

  # ----------------------------------------------
  torch.backends.cudnn.benchmark = True

  #symmetrizer = Symmetrizer()
  print ('# start training', file=sys.stderr)
  soft_mask=None

  #------------- checkpoint set-up + optional resume (upgrade #1) -------------
  checkpoint_models = {'efe': Encoder_Feature_Extractor,
                       'e3dp': Encoder_3D_Predictor,
                       'dec': Decoder_3D}
  checkpoint_optimizers = {'efe': Encoder_Feature_Extractor_Optimer,
                           'e3dp': Encoder_3D_Predictor_Optimer,
                           'dec': Decoder_3D_Optimer}
  if args.downsample_train:
    checkpoint_models['efe_local'] = Encoder_Feature_Extractor_Local
    checkpoint_models['e3dp_local'] = Encoder_3D_Predictor_Local
    checkpoint_optimizers['efe_local'] = Encoder_Feature_Extractor_Optimer_Local
    checkpoint_optimizers['e3dp_local'] = Encoder_3D_Predictor_Optimer_Local
  if args.cascade_refine:
    checkpoint_models['refiner'] = Pose_Refiner
    checkpoint_optimizers['refiner'] = Pose_Refiner_Optimer

  start_epoch = 0
  if args.resume:
    if os.path.isfile(args.resume):
      start_epoch = load_checkpoint(args.resume, checkpoint_models, checkpoint_optimizers, scaler, device)
      print('# resumed from %s -> starting at epoch %d' % (args.resume, start_epoch), file=sys.stderr)
    else:
      print('# --resume given but file not found: %s (starting fresh)' % args.resume, file=sys.stderr)

  #------------- Gaussian spectral (volume) prior: radial frequency weight (upgrade #3) -------------
  # |k|/k_Nyquist in [0,1] for a centered 2D Fourier slice; DC at the box centre.
  slice_freq_radius = (torch.sqrt(helper_x**2 + helper_y**2) / (imgsize / 2.0)).to(device).float()
  volume_prior_weight_map = None
  if args.volume_prior != 'none' and args.volume_prior_weight > 0:
    if args.volume_prior == 'gaussian':
      # w(k) = |k/kNyq|^power : Gaussian/Wiener prior, shrinks amplitude more where
      # the expected signal power is low (high frequency).
      volume_prior_weight_map = torch.pow(slice_freq_radius, args.volume_prior_power)
    elif args.volume_prior == 'exponential':
      # w(k) = exp(beta*|k/kNyq|) - 1 : cryoSPARC-style exponential/B-factor envelope,
      # penalizes the top shells far harder than the power law -> stronger noise-shelf kill.
      volume_prior_weight_map = torch.exp(args.volume_prior_beta * slice_freq_radius) - 1.0
    # optionally zero the prior inside the resolved band so it only bites the noise band
    if args.volume_prior_start_radius > 0:
      volume_prior_weight_map = volume_prior_weight_map * (slice_freq_radius >= args.volume_prior_start_radius).float()
    # normalize so --volume_prior_weight has a comparable meaning across profiles/box sizes
    volume_prior_weight_map = volume_prior_weight_map / (volume_prior_weight_map.mean() + 1e-8)

  # torch.autograd.set_detect_anomaly(True)
  start_res=args.start_resolution  
  end_res=max(args.end_resolution,2*data_apix)


  start_radius=1. / (start_res/data_apix) if start_res>0 else 6./imgsize
  end_radius  = 1. / (end_res/data_apix)

  rescale_step=(end_radius-start_radius) /(args.mask_steps)
  current_free_mem = get_global_min_free_mem_gb(device)
  #==================================#
  for epoch in range(start_epoch, args.num_epochs):
    current_trunk=1
    sampler.set_epoch(epoch)

    # ---- per-epoch LR schedule ---- (cur_lr is shown in the progress line below)
    # Determine the marching frontier + whether the box-228 local-refine stage has engaged, so the
    # LOCAL optimizers can WARM-RESTART their LR from the transition (fade_in_start_epoch) instead of
    # inheriting the already-decayed global LR. Previously the local stage engaged late (LR ~2e-5),
    # then was further scaled by local_refine_scale (0.3) -> effective ~6e-6 on a freshly-initialised
    # encoder -> inert refine -> resolution capped at the coarse (~4A) level.
    current_reconstruct_radius=min(imgsize//2,min(args.mask_steps,(int(epoch/args.mask_epoch))*rescale_step+start_radius)*imgsize)
    if args.downsample_train and current_reconstruct_radius>args.downsample_radius and fade_in_start_epoch==0:
      fade_in_start_epoch=epoch
    cur_lr = lr; local_lr = lr
    if args.lr_schedule != 'none':
      cur_lr = lr_at_epoch(epoch, lr, args.min_lr, args.num_epochs,
                           args.lr_warmup_epochs, args.lr_warmup, args.lr_schedule)
      for _opt in [Encoder_Feature_Extractor_Optimer, Encoder_3D_Predictor_Optimer, Decoder_3D_Optimer]:
        for _pg in _opt.param_groups:
          _pg['lr'] = cur_lr
      if args.cascade_refine:
        for _pg in Pose_Refiner_Optimer.param_groups:
          _pg['lr'] = cur_lr
      if args.downsample_train:
        # warm restart: treat the local stage as its own run starting at fade_in_start_epoch
        # (fresh warmup + cosine decay over the REMAINING epochs). No local_refine_scale on the LR
        # -- that scale now applies ONLY to the correction magnitude (z_3d_local*local_refine_scale).
        if fade_in_start_epoch > 0:
          _lt = max(1, args.num_epochs - fade_in_start_epoch)
          local_lr = lr_at_epoch(epoch - fade_in_start_epoch, lr, args.min_lr, _lt,
                                 args.lr_warmup_epochs, args.lr_warmup, args.lr_schedule)
        else:
          local_lr = cur_lr
        for _opt in [Encoder_Feature_Extractor_Optimer_Local, Encoder_3D_Predictor_Optimer_Local]:
          for _pg in _opt.param_groups:
            _pg['lr'] = local_lr

    #loss_3d=0
    c=0
    N = len(sampler)

    # Tier 0: torch.profiler over the first --profile_iters iterations of --profile_epoch (rank 0 only).
    _prof = None; _prof_count = 0
    if args.profile and epoch == args.profile_epoch and local_rank == 0:
      torch.cuda.reset_peak_memory_stats()
      _prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                 torch.profiler.ProfilerActivity.CUDA],
                                     profile_memory=True)
      _prof.__enter__()

    current_reconstruct_radius=min(imgsize//2,min(args.mask_steps,(int(epoch/args.mask_epoch))*rescale_step+start_radius)*imgsize)

    current_reconstruct_resolution=max(data_apix*2,imgsize*data_apix/current_reconstruct_radius)


    mask_index=min(int(epoch/args.mask_epoch),args.mask_steps-1)
    # this mask is applied on reconstructed images\input_coord grid. (also input ft image)

    if args.downsample_train and current_reconstruct_radius>args.downsample_radius:
      if fade_in_start_epoch==0 :
        fade_in_start_epoch=epoch

      Encoder_Feature_Extractor.eval()     
      Encoder_3D_Predictor.eval()       
      if epoch - fade_in_start_epoch < args.fade_in_epochs:
        Decoder_3D.eval()    
      else:
        Decoder_3D.train()
    else:
      if epoch <= args.lock_epoch:
        Encoder_Feature_Extractor.train()     
      else:
        Encoder_Feature_Extractor.eval() 
      Encoder_3D_Predictor.train()           
      Decoder_3D.train()

    current_mask=fourier_masks[mask_index]    

    if current_free_mem < args.free_gpu_memory : 
      current_dataloader_index+=1
    current_minibatchsize=minibatch_slots[current_dataloader_index]
    train_iterator=train_iterator_list[current_dataloader_index]


    for data_unit in train_iterator:

      #with torch.cuda.amp.autocast():
      with torch.cuda.amp.autocast():
        if len(data_unit)==2:
          y,ctf=data_unit

        elif len(data_unit)==3:
          y,ctf,pose=data_unit
          pose=pose.to(device)        
        elif len(data_unit)==4:
          y,ctf,ctf_params,particles_name=data_unit
        elif len(data_unit)==5:
          y,ctf,pose,ctf_params,particles_name=data_unit   
          pose=pose.to(device) 
        # ---------get data---------
        y=Variable(y)
        y=y.to(device)
        #print(y.size())
        y=y-y.mean(dim=(1, 2), keepdim=True)

        ctf=ctf.to(device)       
        ctf_max=ctf.max()
        ctf_min=ctf.mean()
        train_y=y
        trunk_size=y.size(0)
        accum_iter=args.batch_size//trunk_size
        data_block_size=train_y.size(0)
        train_y=train_y.view(data_block_size,1,imgsize,imgsize)
        if args.skip_read_pose:
          if args.lowpass_input:
            train_y=ift2( torch.mul(ft2(train_y),current_mask))
          # downsample input_image for Encoder .
          if args.downsample_train  :
            downsampled_train_y=downsample_input(train_y,args.downsample_radius,args.downsample_soft_crop)
            #downsampled_train_y=ift2(crop_tensor(ft2(train_y),args.downsample_radius) )
            # downsampled_ctf=crop_tensor(ctf.view(data_block_size,1,imgsize,imgsize),args.downsample_radius)
          #'''
          # -----calculate noise std -----.
          #noise_images=torch.mul(y,noise_mask).view(data_block_size,imgsize,imgsize)
          #noise_ft=ft2(noise_images)
          noise_grad=torch.ones(imgsize,imgsize,device=device)
          # signal_mu=torch.ones(imgsize,imgsize,device=device)
          # for mask_i in range(args.mask_steps):
          #   # current_noise_std=1.414*torch.mean (torch.std(noise_ft[new_fourier_masks[mask_i].expand( data_block_size,imgsize,imgsize)>0.49 ]) ,dim=-1)
          #   # print(current_noise_std)   
          #   noise_grad[new_fourier_masks[mask_i]>0.1]=(mask_i*args.DW_factor+2*args.mask_steps)/(2*args.mask_steps)
          #   # signal_mu[new_fourier_masks[mask_i]>0.49]=torch.mean(torch.mean(train_y.view(data_block_size,imgsize,imgsize) [new_fourier_masks[mask_i].expand(data_block_size,imgsize,imgsize)>0.49 ]) ,dim=0)
          # #'''
          if args.gaussian or args.grad:
            train_y=pre_module(train_y)
          elif args.preprocess:
            train_y=torch_apply_lowpass(train_y,args.preprocessing_output_channels,pre_module)
            if args.downsample_train:
              downsampled_train_y=torch_apply_lowpass(downsampled_train_y,args.preprocessing_output_channels,downsampled_pre_module)
          if args.sym:
            degrees=np.random.rand(args.sym_factor)*360 if args.random_augmentation else []
            #print(train_y.size())
            train_y=image_duplicator(train_y,args.sym_factor,args.rot_manner,degrees)
            #print(train_y.size())          
            ctf=image_duplicator(ctf,args.sym_factor,args.rot_manner,degrees)        
            y=image_duplicator(y,args.sym_factor,args.rot_manner,degrees)
            if args.downsample_train  :
              downsampled_train_y=image_duplicator(downsampled_train_y,args.sym_factor,args.rot_manner,degrees)

          #  b = sym_factor * data_block_size   
          b=train_y.size(0)
          # New pre module should downsample particles together
          train_y=torch.mul(train_y,input_mask)
          if args.downsample_train:
            if current_reconstruct_radius>args.downsample_radius  :
              Encoder_Feature_Extractor.eval()
              Encoder_3D_Predictor.eval()       
              features_local=enc_fwd(Encoder_Feature_Extractor_Local, train_y, args)
              features=enc_fwd(Encoder_Feature_Extractor, downsampled_train_y, args)
            else:
              if args.random_input:
                downsampled_train_noise=torch.rand_like(downsampled_train_y)
                features=enc_fwd(Encoder_Feature_Extractor, downsampled_train_noise, args)
              else:
                features=enc_fwd(Encoder_Feature_Extractor, downsampled_train_y, args)
          else:
            if args.random_noise:
              features=enc_fwd(Encoder_Feature_Extractor, train_y+torch.randn_like(train_y), args)
            else:
              features=enc_fwd(Encoder_Feature_Extractor, train_y, args)

          z_3d=Encoder_3D_Predictor(features) 
          if args.downsample_train and current_reconstruct_radius>args.downsample_radius :
            z_3d_local=Encoder_3D_Predictor_Local(features_local)
            z_3d=z_3d+z_3d_local*args.local_refine_scale

          rotation_3d=z_3d[:,:6]

          # add random distrubance to escape from  bad initilization in early epochs
          if args.add_distrubance and epoch < args.distrubance_epoch:
            #distrubance_ratio=args.distrubance_ratio-epoch*(args.distrubance_ratio/ args.distrubance_epoch)
            distrubance_ratio=cosine_decay(epoch,args.distrubance_epoch,args.distrubance_ratio,args.distrubance_end_ratio)
            rotation_3d=rotation_3d + torch.randn_like(rotation_3d)* distrubance_ratio if args.gaussian_distrubance else rotation_3d + (torch.rand_like(rotation_3d)-0.5)*distrubance_ratio


          if args.latent_code_dim_3d!=0:
            latent_3d=z_3d[:,6:6+args.latent_code_dim_3d]  
          translation_3d=z_3d[:,-2:]*args.dx_scale*imgsize        
          #-----------apply rotation on input coords
          center_plane_grid =  x_coord_plane.to(device)
          #center_plane_grid_triu=torch.triu(center_plane_grid.view(imgsize,imgsize,3))
          center_plane_grid =  center_plane_grid.expand(b,center_plane_grid.size(0),center_plane_grid.size(1))
          norm_rotation_3d=sixd_to_two_orthonormal(rotation_3d)
          if args.norm6d:            
            rot_3d_matrix=transforms.rotation_6d_to_matrix(norm_rotation_3d)
          else:
            rot_3d_matrix=transforms.rotation_6d_to_matrix(rotation_3d)    
          rot_uniform_loss=uniform_s2_moment_loss(rot_3d_matrix,use_all_axes=True)
        else:
          rot_3d_matrix=transforms.euler_angles_to_matrix(pose,"ZYZ")
        if mode=='3d' or mode=='both':
          #rotated_plane_grid=torch.bmm(center_plane_grid,rot_3d_matrix)
          rotated_plane_grid=torch.bmm(center_plane_grid,rot_3d_matrix.transpose(1,2))
          rotated_plane_grid=rotated_plane_grid.view(b,imgsize,imgsize,3)

          y=torch.mul(reconstruct_mask,y)

          #-----------decoder
          if args.reduce_input:
              rotated_plane_grid=rotated_plane_grid[current_mask.expand(b,imgsize,imgsize)>0.49]   
              pred_images_3d=torch.zeros(b,imgsize,imgsize,2,device=device)              

          decoder_output=dec_fwd(Decoder_3D, rotated_plane_grid, latent_3d if args.latent_code_dim_3d!=0 else None, args)

          if args.reduce_input:
            pred_images_3d[current_mask.expand(b,imgsize,imgsize)>0.49]=decoder_output.view(-1,2)  
          else:
            pred_images_3d=decoder_output

          #-----------apply translation and CTF on pred images
          pred_images_3d = torch.view_as_complex(pred_images_3d)

          pred_images_3d=phase_shift_ft(pred_images_3d,translation_3d,helper_x,helper_y)

          # ---- Gaussian spectral (volume) prior (upgrade #3) ----
          # Penalize predicted Fourier amplitude, shell-weighted so high frequencies
          # (where expected signal power is low) are shrunk toward zero. This is the
          # RELION-style Gaussian/Wiener prior; it directly damps the high-frequency
          # noise shelf. Computed on the decoded structure-factor slice, before the
          # real-space apodization remask and before CTF.
          if volume_prior_weight_map is not None:
            # compute in fp32: the (unclamped) exp(envelope) decoder can spike, which would
            # overflow an fp16 square and poison the loss with inf.
            _pr = pred_images_3d.real.float()
            _pi = pred_images_3d.imag.float()
            volume_prior_loss = args.volume_prior_weight * torch.mean(
                volume_prior_weight_map * (_pr*_pr + _pi*_pi))
          else:
            volume_prior_loss = None

          pred_images_3d=ft2(torch.mul(ift2(pred_images_3d),reconstruct_mask  ) ) if soft_mask is None else ft2(torch.mul(ift2(pred_images_3d),binary_mask  ) )

          ctf=ctf.view(b,imgsize,imgsize)
          pred_images_3d=torch.mul(pred_images_3d,ctf)


          if args.shift_ori:
            #print(b,translation_3d.size(0),y.size(0))
            shifted_mask=ift2(phase_shift_ft(ft2(reconstruct_mask.view(1,imgsize,imgsize).expand(b,imgsize,imgsize)),translation_3d,helper_x,helper_y))
            shifted_mask=(shifted_mask>0.5).float()
            #DL.write_file(dt(shifted_mask),args.render_path+'/shifted_mask_'+str(epoch+1)+'.mrc')
            shifted_mask=shifted_mask.view(b,1,imgsize,imgsize)
            #print(y.size(),shifted_mask.size())
            #y=torch.mul(reconstruct_mask,y)
            y=torch.mul(shifted_mask,y)
          # ------------ loss------------------
          y_ft=ft2(y.view(b,imgsize,imgsize))
          # y_ft=torch_highpass(y_ft)

          y_ft=torch.mul(y_ft,current_mask)

          pred_images_3d=torch.mul(pred_images_3d,current_mask)

          # ---- Option-B conditioned cascade: render-and-compare pose refinement (K=1) ----
          # pred_images_3d above is the COARSE render (at rot_3d_matrix, translation_3d), already
          # masked. The residual (y - pred@coarse) directly encodes the pose error; a zero-init
          # refiner predicts a small so(3)+shift update; re-render at the refined pose.
          cascade_aux_loss = None
          if args.cascade_refine:
            def _render(_R, _t):
              _grid = torch.bmm(center_plane_grid, _R.transpose(1,2)).view(b,imgsize,imgsize,3)
              if args.reduce_input:
                _sel = current_mask.expand(b,imgsize,imgsize) > 0.49
                _dec = dec_fwd(Decoder_3D, _grid[_sel], latent_3d if args.latent_code_dim_3d!=0 else None, args)
                _p = torch.zeros(b,imgsize,imgsize,2,device=device)
                _p[_sel] = _dec.view(-1,2)
              else:
                _p = dec_fwd(Decoder_3D, _grid, latent_3d if args.latent_code_dim_3d!=0 else None, args)
              _p = torch.view_as_complex(_p)
              _p = phase_shift_ft(_p, _t, helper_x, helper_y)
              if volume_prior_weight_map is not None:
                _prr = _p.real.float(); _pii = _p.imag.float()
                _vp = args.volume_prior_weight * torch.mean(volume_prior_weight_map*(_prr*_prr+_pii*_pii))
              else:
                _vp = None
              _p = ft2(torch.mul(ift2(_p), reconstruct_mask))
              _p = torch.mul(_p, ctf.view(b,imgsize,imgsize))
              _p = torch.mul(_p, current_mask)
              return _p, _vp
            # detach: the refiner treats the residual as a FIXED observed error (a learned
            # optimizer step reads the residual, it doesn't backprop through the coarse render).
            # Also avoids the DDP "mark ready twice" error from Decoder_3D getting grad from both
            # the coarse and refined decode (with --cascade_lambda 0 the decoder is then in the graph
            # only via the refined render). NOTE: --cascade_lambda>0 needs ddp_wrap find_unused_parameters=True.
            _resid_img = ift2(y_ft - pred_images_3d.detach()).real.view(b,1,imgsize,imgsize)
            if args.channels_last:
              _resid_img = _resid_img.contiguous(memory_format=torch.channels_last)
            _dp = Pose_Refiner(_resid_img)
            _dw = _dp[:, :3].float()
            _dt = _dp[:, 3:5]
            _dR = transforms.axis_angle_to_matrix(_dw).to(rot_3d_matrix.dtype)
            _R1 = torch.bmm(_dR, rot_3d_matrix)
            _t1 = translation_3d + _dt * args.cascade_shift_scale
            if args.cascade_lambda > 0:
              _cl, _ = symmetric_complex_loss(pred_images_3d, y_ft, args.sym_factor)
              cascade_aux_loss = args.cascade_lambda * _cl.mean()
            pred_images_3d, volume_prior_loss = _render(_R1, _t1)

          # noise grad could be seen as weighted on different resolution.
          # test no noise_grad,this make loss larger
          # 
          if args.DW:

            w_shell = make_shell_weights_midband(
              mask_steps=args.mask_steps,
              device=noise_grad.device,
              dtype=noise_grad.dtype
            )

            noise_grad = torch.ones_like(noise_grad) * w_shell[0]  
            for mask_i in range(args.mask_steps):
              # noise_grad[new_fourier_masks[mask_i]>0.1]=(mask_i*args.DW_factor+2*args.mask_steps)/(2*args.mask_steps)
              noise_grad[new_fourier_masks[mask_i]>0.1] = w_shell[mask_i]

            y_ft=torch.mul(y_ft,noise_grad)
            pred_images_3d=torch.mul(pred_images_3d,noise_grad)

          if args.reduce_input:
            y_ft=y_ft[current_mask.expand(b,imgsize,imgsize)>0.49].view(b,-1)
            pred_images_3d=pred_images_3d[current_mask.expand(b,imgsize,imgsize)>0.49].view(b,-1)   


          if  args.W_loss:
            loss_3d,active_path= symmetric_complex_wloss(pred_images_3d,y_ft,args.sym_factor)
          elif args.soft_min:
            loss_3d,active_path= symmetric_complex_softloss(pred_images_3d,y_ft,args.sym_factor,args.soft_min_temp)
          else:
            if args.L1:
              loss_3d,active_path= symmetric_complex_L1_loss(pred_images_3d,y_ft,args.sym_factor)
            else:
              loss_3d,active_path= symmetric_complex_loss(pred_images_3d,y_ft,args.sym_factor)

          if args.save_pose and (epoch+1)%args.save_interval==0 :
            newfile_marker=False
            out_pose_name=args.render_path+'//'+args.output_pose_file+'_'+str(epoch+1)+f'_rank{rank}.star'
            if not os.path.exists(out_pose_name) and local_rank==args.write_device:
              newfile_marker=True
              pre_append='data_particles\n\nloop_\n_rlnImageName #1\n_rlnAngleRot #2\n_rlnAngleTilt #3\n_rlnAnglePsi #4\n_rlnDefocusU #5\n_rlnDefocusV #6\n_rlnDefocusAngle #7\n_rlnVoltage #8\n_rlnSphericalAberration #9\n_rlnAmplitudeContrast #10\n_rlnOriginX #11\n_rlnOriginY #12\n_rlnImagePixelSize #13\n_rlnrotation6d #14\n_rlnnorm6d\n'

            output_pose_file=open(out_pose_name,'a+')
            if newfile_marker:
              output_pose_file.write(pre_append)
            #predicted_pose_euler=transforms.matrix_to_euler_angles(transforms.rotation_6d_to_matrix(rotation_3d[active_path]),'ZYZ')
            if args.norm6d:
              temp_r=Rotation.from_matrix(transforms.rotation_6d_to_matrix(norm_rotation_3d[active_path]).cpu().detach().numpy())
            else:
              temp_r=Rotation.from_matrix(transforms.rotation_6d_to_matrix(rotation_3d[active_path]).cpu().detach().numpy())
            predicted_pose_euler=temp_r.as_euler('zyz', degrees=True)


            for particle_write_index in range(len(particles_name)):
              #print(torch.isinf(rotation_3d).any())
              temp_rotation_3d=np.round(rotation_3d.cpu().detach().float().numpy(),5)
              #print(np.isinf(temp_rotation_3d).any())
              temp_norm_rotation_3d=np.round(norm_rotation_3d.cpu().detach().float().numpy(),5)
              if np.isinf(temp_rotation_3d).any():
                current_pose_line="inf \n"
              else:
                current_pose_line=particles_name[particle_write_index]+' '+str(predicted_pose_euler[particle_write_index][0])+' '+str(predicted_pose_euler[particle_write_index][1])+' '+str(predicted_pose_euler[particle_write_index][2])+' '+str(ctf_params['dfu'][particle_write_index].numpy())+' '+str(ctf_params['dfv'][particle_write_index].numpy())+' '+str(ctf_params['dfang'][particle_write_index].numpy())+' '+str(ctf_params['volt'][particle_write_index].numpy())+' '+str(ctf_params['cs'][particle_write_index].numpy())+' '+str(ctf_params['ampcont'][particle_write_index].numpy())+' '+str(translation_3d.cpu().detach().numpy()[particle_write_index][0])+' '+str(translation_3d.cpu().detach().numpy()[particle_write_index][1])+' '+str(ctf_params['apix'][particle_write_index].numpy())+' '+str(temp_rotation_3d[particle_write_index])+' '+str(temp_norm_rotation_3d[particle_write_index])+'  \n'

              output_pose_file.write(current_pose_line)
            output_pose_file.flush()
            output_pose_file.close()


          if args.hard_topk and epoch >= args.topk_start_epoch and epoch < args.topk_end_epoch:
            current_ratio=((args.topk_end_ratio-args.topk_start_ratio)/(args.topk_end_epoch-args.topk_start_epoch))*(epoch-args.topk_start_epoch)+args.topk_start_ratio
            loss_3d=torch.topk(loss_3d,int(data_block_size*current_ratio),largest=False).values  
          elif args.soft_topk:
            loss_3d_mu=loss_3d.mean()
            loss_3d_std=loss_3d.std()
            loss_3d=loss_3d[loss_3d.ge(loss_3d_mu-args.sigma_level*loss_3d_std)*loss_3d.le(loss_3d_mu+args.sigma_level*loss_3d_std)]
            current_ratio=loss_3d.size(0)/data_block_size

          # loss_3d=torch.add(loss_3d,output_loss*400)

          loss_3d=torch.mean(loss_3d)
          loss_3d=torch.div(loss_3d,accum_iter)
          if args.uniform and epoch < args.uniform_epoch:
            loss_3d=torch.add(loss_3d,torch.div(rot_uniform_loss,accum_iter))  # scale by accum_iter like the data term + volume prior (was accum_iter x too strong)
          if volume_prior_loss is not None:
            loss_3d=torch.add(loss_3d, torch.div(volume_prior_loss, accum_iter))
          if cascade_aux_loss is not None:
            loss_3d=torch.add(loss_3d, torch.div(cascade_aux_loss, accum_iter))  # optional aux loss on the coarse render (--cascade_lambda)


          if trunk_size*current_trunk>=args.batch_size:
            update_grad=True
            current_trunk=1
            # loss_3d=torch.div(loss_3d,accum_iter)

          else:
            current_trunk+=1
            update_grad=False

          if mode=='3d' :
            if args.cascade_refine:
              Pose_Refiner_Optimer.zero_grad()
            if args.downsample_train :
             
              #if epoch <  args.downsample_train_epoch  :
              if current_reconstruct_radius <  args.downsample_radius  :              
                Encoder_Feature_Extractor_Optimer.zero_grad()
                Encoder_3D_Predictor_Optimer.zero_grad()  
                Decoder_3D_Optimer.zero_grad() 
              else:
                Encoder_Feature_Extractor_Optimer_Local.zero_grad()
                Encoder_3D_Predictor_Optimer_Local.zero_grad()
                # fade-in freeze: for the first --fade_in_epochs after local refinement
                # engages (radius first exceeds downsample_radius -> fade_in_start_epoch),
                # freeze the global model + decoder so the fresh local encoder can warm up.
                if epoch >= fade_in_start_epoch + args.fade_in_epochs :
                  if not args.local_only:
                    Encoder_Feature_Extractor_Optimer.zero_grad()
                    Encoder_3D_Predictor_Optimer.zero_grad()
                  Decoder_3D_Optimer.zero_grad()

            else:
              if epoch <= args.lock_epoch:
                Encoder_Feature_Extractor_Optimer.zero_grad()
              Encoder_3D_Predictor_Optimer.zero_grad()                
              Decoder_3D_Optimer.zero_grad()    


          # Tier 1: skip DDP gradient all-reduce on non-final accumulation micro-steps.
          # no_sync accumulates grads locally; the final micro-step (update_grad) all-reduces
          # once. Mathematically identical to per-step sync, just far less communication.
          if args.fast_ddp and not update_grad:
            _ddp_models = [Encoder_Feature_Extractor, Encoder_3D_Predictor, Decoder_3D]
            if args.downsample_train:
              _ddp_models += [Encoder_Feature_Extractor_Local, Encoder_3D_Predictor_Local]
            if args.cascade_refine:
              _ddp_models += [Pose_Refiner]
            with contextlib.ExitStack() as _stack:
              for _m in _ddp_models:
                _stack.enter_context(_m.no_sync())
              scaler.scale(loss_3d).backward()
          else:
            scaler.scale(loss_3d).backward()

          if update_grad :
            if mode=='3d':
              if args.downsample_train:
                if current_reconstruct_radius <  args.downsample_radius  : 

                  scaler.step(Encoder_Feature_Extractor_Optimer) 
                  scaler.step(Encoder_3D_Predictor_Optimer)
                  scaler.step(Decoder_3D_Optimer)   
                else:
                  scaler.step(Encoder_Feature_Extractor_Optimer_Local)
                  scaler.step(Encoder_3D_Predictor_Optimer_Local)
                  if epoch  >= fade_in_start_epoch + args.fade_in_epochs :
                    if not args.local_only:
                      scaler.step(Encoder_Feature_Extractor_Optimer)
                      scaler.step(Encoder_3D_Predictor_Optimer)
                    scaler.step(Decoder_3D_Optimer)
              else:
                #if epoch%2==0:
                if epoch <= args.lock_epoch:
                  scaler.step(Encoder_Feature_Extractor_Optimer) 
                scaler.step(Encoder_3D_Predictor_Optimer)
                #else:
                scaler.step(Decoder_3D_Optimer)
            # else:
            #   scaler.step(Encoder_Feature_Extractor_Optimer_3d)
            if args.cascade_refine:
              scaler.step(Pose_Refiner_Optimer)

            scaler.update()


        c+=data_block_size
        eclipsed_time=time.time()-start_time
        if mode=='2d':
          template = '# [{}/{}] training {:.1%},  Error_2d={:.4f}' 
          line = template.format(epoch, num_epochs, c/N, loss_2d.item())
        elif mode=='3d':
          template = '# [{}/{}] time: {:.1f} s, training {:.1%},  Resolution : {:.1f}, Radius : {:.1f}, Minibatch : {}, lr={:.2e}, Error_3d={:.4f} , Uniform={:.3f}'
          line = template.format(epoch, num_epochs, eclipsed_time, c/N, current_reconstruct_resolution , current_reconstruct_radius, current_minibatchsize, cur_lr, loss_3d.item()*accum_iter,rot_uniform_loss.item())
          # template = '# [{}/{}] time: {:.1f} s, training {:.1%},  Error_3d={:.4f},  Current_ratio={:.2f}' 
          # line = template.format(epoch, num_epochs, eclipsed_time, c/N, loss_3d.item()*accum_iter, current_ratio)          
        elif mode=='both':
          if args.consistance_loss:
            template = '# [{}/{}] training {:.1%},  Error_2d={:.4f} ,  Error_3d={:.4f}, Error_consis={:.4f}' 
            line = template.format(epoch, num_epochs, c/N, loss_2d.item(),loss_3d.item(),loss_consis.item())   
          else:
            template = '# [{}/{}] training {:.1%},  Error_2d={:.4f} ,  Error_3d={:.4f}' 
            line = template.format(epoch, num_epochs, c/N, loss_2d.item(),loss_3d.item())    
        print(line,"\r", end="", flush=True)
        if _prof is not None:
          _prof_count += 1
          if _prof_count >= args.profile_iters:
            _prof.__exit__(None, None, None)
            print("\n# ===== TIER-0 PROFILE  (top CUDA ops over %d iters) =====" % args.profile_iters, file=sys.stderr)
            print(_prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25), file=sys.stderr, flush=True)
            print("\n# ===== TIER-0 MEMORY  (top ops by self CUDA memory) =====", file=sys.stderr)
            try:
              print(_prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=25), file=sys.stderr, flush=True)
            except Exception as _me:
              print("# memory table failed: %s" % _me, file=sys.stderr)
            print("# peak CUDA mem this rank: %.2f GB (max_memory_allocated over the profiled window)"
                  % (torch.cuda.max_memory_allocated() / 1e9), file=sys.stderr, flush=True)
            try:
              _trace = os.path.join(args.render_path, "profile_trace.json")
              _prof.export_chrome_trace(_trace)
              print("# chrome trace -> %s" % _trace, file=sys.stderr, flush=True)
            except Exception as _e:
              print("# trace export failed: %s" % _e, file=sys.stderr, flush=True)
            _prof = None
    current_free_mem = get_global_min_free_mem_gb(device)
    torch.cuda.empty_cache()          

    if (epoch+1)%args.save_interval==0:

      distr.barrier(group=sync_pg)


    if (epoch+1)%args.save_interval==0 and rank==args.write_device :   

      y=None
      pred_images_3d=None


      _render_efe = _bare(Encoder_Feature_Extractor)
      _render_e3dp = _bare(Encoder_3D_Predictor)
      _render_dec = _bare(Decoder_3D)

      _render_efe.eval()
      _render_e3dp.eval()
      _render_dec.eval()
      with torch.no_grad():
        if latent_code_dim_2d !=0:

          test_data_unit=next(iter(train_iterator))            
          if len(test_data_unit)==2:
            y,ctf=test_data_unit

          elif len(test_data_unit)==3:
            y,ctf,pose=test_data_unit
            pose=pose.to(device)          
          y=y.to(device)
          y=y[:render_number,:].view(-1,1,imgsize,imgsize)            
          if args.gaussian:
            y=pre_module(y)
          elif args.preprocess:
            y=torch_apply_lowpass(y,args.preprocessing_output_channels,pre_module)
        if mode=='3d':



          coords_3d=x_coord_3d.to(device)
          # symmetrizer.initialize(coords_3d)
          # coords_3d=symmetrizer.symmetrize_input(coords_3d)
          #coords_3d=coords_3d.expand(args.render_number,coords_3d.size(0),coords_3d.size(1)) if args.render_number >1 else coords_3d.view(imgsize,imgsize,imgsize,3)
          coords_3d=coords_3d.view(imgsize,imgsize,imgsize,3)          
          latent_code_3d=None
          test_pred_images_3D=torch.zeros(imgsize,imgsize,imgsize,2,device=device)

          if latent_code_dim_3d!=0:
            latent_code_3d=_render_e3dp(_render_efe(y))[:,6:latent_code_dim_3d+6]
          for slicei in range(imgsize):
            test_pred_images_3D[slicei:slicei+1,...]=_render_dec(coords_3d[slicei:slicei+1,...]) if latent_code_3d is None else _render_dec(coords_3d[slicei:slicei+1,...],latent_code_3d) 
        


            

          test_pred_images_3D=torch.view_as_complex(test_pred_images_3D)
          test_pred_images_3D[imgsize//2,imgsize//2,imgsize//2]=0
          # if args.ignore_DC_component:
          test_pred_images_3D[imgsize//2,imgsize//2,imgsize//2]=0



            

          #test_pred_images_3D=symmetrize_output(test_pred_images_3D)
          # do not use 3dmask to escape from rings.(20250305)
          # but for hard cases this will largely damage output quality.
          if args.lowpass:
            # low-pass the render at the TRAINING FRONTIER (current_reconstruct_radius), as a 3D
            # SPHERICAL soft mask. Previously this used a faster-growing 2D mask
            # ((step+1)*0.5/mask_steps -> overshot the frontier, e.g. r73 vs r47 at ep120) which
            # displayed untrained SIREN extrapolation beyond the constrained band as broken high-freq
            # speckle (worst where density is highest -> the particle centre). The 2D default also left
            # one axis unfiltered. Spherical + at-frontier + soft edge removes it (and the old rings).
            mask_3d=gen_mask(imgsize, current_reconstruct_radius/imgsize, '3d', soft_edge=max(_soft_edge,0.02) ).to(device)
            test_pred_images_3D=torch.mul(test_pred_images_3D,mask_3d)

          # generate a new soft here.
          #  def gen_volume_mask(input_tensor,mask_lowpass=20,mask_lowpass_type='butterworth',mask_lowpass_threhold=2,mask_expandsize=3,mask_soft_expand=12):
          #print('generating soft mask...')
          # if epoch+1 >= args.mask_start_epoch:
          #   soft_mask=gen_volume_mask(test_pred_images_3D.real,lowpass_mask_3d,mask_threshold=args.mask_lowpass_threshold,mask_dilation_width=args.mask_dilation_width,mask_padding_width=args.mask_padding_width).to(device) 
          #   DL.write_file(dt(soft_mask),args.render_path+'/rec_3dmask_'+str(epoch+1)+'.mrc')


          test_pred_images_3D=ift3(test_pred_images_3D).real
          test_pred_images_3D=torch.mul(test_pred_images_3D,reconstruct_mask_3d)

          # ---- output normalization (upgrade #5) ----
          # zero-mean/unit-std inside the reconstruction mask instead of global min-max
          # (so one hot voxel no longer sets the scale and solvent sits near 0).
          if args.norm_zscore:
            _sel = reconstruct_mask_3d > 0.5
            if _sel.sum() > 0:
              _mu = test_pred_images_3D[_sel].mean()
              _sd = test_pred_images_3D[_sel].std()
            else:
              _mu = test_pred_images_3D.mean()
              _sd = test_pred_images_3D.std()
            test_pred_images_3D=dt((test_pred_images_3D-_mu)/(_sd+1e-8))
          else:
            test_pred_images_3D=DL.norm_image(dt(test_pred_images_3D))

          # ---- non-negativity on real-space density (upgrade #4) ----
          # Electron density is physically >=0; clamp LAST so the written map is truly >=0.
          # (Must follow z-score: z-score subtracts the in-mask mean, which would otherwise
          #  re-introduce negative solvent. Clamping here floors solvent at 0, keeps density.)
          if args.enforce_non_neg:
            test_pred_images_3D=np.clip(test_pred_images_3D,0,None)

          # write with the pixel size stamped in the header (plain write_file left apix=0)
          DL.write_file_apix(test_pred_images_3D,args.render_path+'/rec_3d_'+str(epoch+1)+'_'+str(int(eclipsed_time/60))+'min.mrc',data_apix)

          # ---- checkpoint (upgrade #1) ----
          if args.save_checkpoint:
            _ckpt_int = args.checkpoint_interval if args.checkpoint_interval > 0 else args.save_interval
            if (epoch+1) % _ckpt_int == 0:
              save_checkpoint(os.path.join(args.render_path, 'checkpoint.pt'), epoch,
                              checkpoint_models, checkpoint_optimizers, scaler, args)
              print('# saved checkpoint at epoch %d' % (epoch+1), file=sys.stderr)


          test_pred_images_3D=None
          torch.cuda.empty_cache()

    if (epoch+1)%args.save_interval==0:
      distr.barrier(group=sync_pg)
  end_time=time.time()
  time_cost=end_time-start_time
  print("\ntime cost:%.3f s"%time_cost)
       
if __name__ == '__main__':
    main()
