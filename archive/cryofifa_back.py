from __future__ import print_function, division
import warnings
warnings.filterwarnings('ignore')
from new_dataloader import StarDataSet

import numpy as np
import pandas as pd
import sys
from cam_module import *
import gc
from new_dataloader import LmdbDataSet


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

from fairscale.optim.grad_scaler import ShardedGradScaler

from torch.profiler import profile




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

  local_rank=int(os.environ["LOCAL_RANK"])
  setup_for_distributed(local_rank==0)
  torch.cuda.set_device(local_rank)
  distr.init_process_group("nccl",world_size=args.world_size,rank=local_rank)
  device=torch.device('cuda')  

  #======================================#
  if not os.path.exists(args.render_path) and (local_rank==0):
    os.mkdir(args.render_path)
  start_time=time.time()

  #=================data loader======================#
   
  data_train,total_len,data_apix= data_constructor(args.data_path,args.particles_meta,imgsize,args.lmdb_path)    
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
    train_iterator_list.append (torch.utils.data.DataLoader(data_train, batch_size=batchsize_slot,shuffle=False,pin_memory=True,num_workers=args.num_workers,drop_last=True,sampler=sampler))
  current_dataloader_index=0
  
  
  #==========preprocessing module=========#
  pre_module=preprocesser(lowpass=True,gaussian=args.gaussian,preprocessing_output_channels=args.preprocessing_output_channels,imgsize=imgsize,kernel_size=11,kernel_variance=0.01,include_input=False)
  pre_module=pre_module.to(device)    

  downsampled_pre_module=preprocesser(lowpass=True,gaussian=args.gaussian,preprocessing_output_channels=args.preprocessing_output_channels,imgsize=2*args.downsample_radius,kernel_size=11,kernel_variance=0.01,include_input=False)

  downsampled_pre_module=downsampled_pre_module.to(device)    

  input_mask=gen_mask(imgsize,args.input_radius).view(imgsize,imgsize).to(device)


  reconstruct_mask=gen_mask(imgsize,args.reconstruct_radius).view(imgsize,imgsize).to(device)
  radwn=int(imgsize//2*data_apix/args.mask_lowpass)
  radwn=16
  lowpass_mask_3d=torch.from_numpy(gen_lowpassmask_3d(imgsize,radwn)).float().to(device)

  fourier_masks,new_fourier_masks=reconstruct_fourier_masks(start_resolution=args.start_resolution,data_apix=data_apix,imgsize=imgsize,downsample_train=False,mask_steps=args.mask_steps,butterworth=False,exp_mask=False,device=device,end_resolution=args.end_resolution)
  noise_mask=gen_mask(imgsize,args.noise_radius,'2d',True).to(device)
  reconstruct_mask_3d=gen_mask(imgsize,args.reconstruct_radius,'3d').to(device)

  #Tensor_croper=torchvision.transforms.CenterCrop(args.downsample_radius*2).to(device)
  #===============models==================#
  shift_dim=2
  fade_in_start_epoch=0  

  lr =args.learning_rate  
  base_optimizer_arguments = { "lr": lr,"eps": 1e-8}
  base_optimizer=torch.optim.Adam
  in_channels=args.preprocessing_output_channels  if args.preprocess else 1
  Encoder_Feature_Extractor = model_constructor(name='Feature_Extrator',backbone=args.encoder_backbone,basename=args.encoder_basename,in_channels=in_channels,image_size=imgsize,use_batchnorm=False)
  #Encoder_Feature_Extractor.apply(weight_init)
  Encoder_Feature_Extractor=Encoder_Feature_Extractor.to(device)
  #***********FAIRSCALE MODEL CONFIG*****************#   
  # Encoder_Feature_Extractor_Optimer=OSS(params=Encoder_Feature_Extractor.parameters(),optim=base_optimizer,**base_optimizer_arguments)
  # Encoder_Feature_Extractor=ShardedDDP(Encoder_Feature_Extractor,Encoder_Feature_Extractor_Optimer)

  Encoder_Feature_Extractor_Optimer=optim.Adam(Encoder_Feature_Extractor.parameters(),lr=lr)


  Encoder_Feature_Extractor=DDP(Encoder_Feature_Extractor,device_ids=[local_rank])



  if args.downsample_train:
    Encoder_Feature_Extractor_Local = model_constructor(name='Feature_Extrator',backbone=args.encoder_backbone,basename=args.encoder_basename,in_channels=in_channels,image_size=imgsize,use_batchnorm=False)
    Encoder_Feature_Extractor_Local=Encoder_Feature_Extractor_Local.to(device)
    Encoder_Feature_Extractor_Optimer_Local=optim.Adam(Encoder_Feature_Extractor_Local.parameters(),lr=lr)
    Encoder_Feature_Extractor_Local=DDP(Encoder_Feature_Extractor_Local,device_ids=[local_rank])
  if args.mode=='both':

    Encoder_Feature_Extractor_3d = model_constructor(name='Feature_Extrator',backbone=args.encoder_backbone,basename=args.encoder_basename,in_channels=args.preprocessing_output_channels+1,image_size=imgsize,use_batchnorm=False)

    Encoder_Feature_Extractor_3d=Encoder_Feature_Extractor_3d.to(device)
    Encoder_Feature_Extractor_Optimer_3d=optim.Adam(Encoder_Feature_Extractor_3d.parameters(),lr=lr)


    Encoder_Feature_Extractor_3d=DDP(Encoder_Feature_Extractor_3d,device_ids=[local_rank])

  if args.mode=='3d' or args.mode=='both':
    rotation_dim=6 

    Encoder_3D_Predictor = model_constructor(name='Encode_Predictor',rotation_dim=rotation_dim,shift_dim=2,latent_code_dim=args.latent_code_dim_3d,activation=args.encoder_activation,last_activation=args.last_activation,hidden_dim=args.encoder_hidden_dim,add_layers=args.encoder_add_layers,use_KAN=args.use_KAN)
    Encoder_3D_Predictor.apply(weight_init)
    Encoder_3D_Predictor=Encoder_3D_Predictor.to(device)  
    Encoder_3D_Predictor_Optimer=optim.Adam(Encoder_3D_Predictor.parameters(),lr=lr)

    Encoder_3D_Predictor=DDP(Encoder_3D_Predictor,device_ids=[local_rank])


    if args.downsample_train:
      Encoder_3D_Predictor_Local = model_constructor(name='Encode_Predictor',rotation_dim=rotation_dim,shift_dim=2,latent_code_dim=args.latent_code_dim_3d,activation=args.encoder_activation,last_activation=args.last_activation,hidden_dim=args.encoder_hidden_dim,add_layers=args.encoder_add_layers)
      Encoder_3D_Predictor_Local.apply(weight_init)
      Encoder_3D_Predictor_Local=Encoder_3D_Predictor_Local.to(device)  
      Encoder_3D_Predictor_Optimer_Local=optim.Adam(Encoder_3D_Predictor_Local.parameters(),lr=lr)

      Encoder_3D_Predictor_Local=DDP(Encoder_3D_Predictor_Local,device_ids=[local_rank])



    Decoder_3D= model_constructor(name='Decoder',hidden_dim=args.decoder_hidden_dim, num_layers=args.decoder_layers,add_layers=args.decoder_add_layers,activation='SIREN',w0=args.w0,w=args.w,out_dim=2,in_dim=3,latent_code_dim=args.latent_code_dim_3d,use_symmetrizer=args.use_symmetrizer)
    Decoder_3D.apply(weight_init)
    Decoder_3D=Decoder_3D.to(device) 
    #***********FAIRSCALE MODEL CONFIG*****************#     
    # Decoder_3D_Optimer=OSS(params=Decoder_3D.parameters(),optim=base_optimizer,**base_optimizer_arguments)     
    # Decoder_3D=ShardedDDP(Decoder_3D,Decoder_3D_Optimer)
    Decoder_3D_Optimer=optim.Adam(Decoder_3D.parameters(),lr=lr)
    Decoder_3D=DDP(Decoder_3D,device_ids=[local_rank])

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

  # torch.autograd.set_detect_anomaly(True)
  start_res=args.start_resolution  
  end_res=max(args.end_resolution,2*data_apix)


  start_radius=1. / (start_res/data_apix) if start_res>0 else 6./imgsize
  end_radius  = 1. / (end_res/data_apix)

  rescale_step=(end_radius-start_radius) /(args.mask_steps)
  #current_free_mem=torch.cuda.mem_get_info(local_rank)[0]/1024/1024/1024
  free_gpus_memorys=np.zeros(args.world_size)
  for gpui in range(args.world_size):
    free_gpus_memorys[gpui]=(torch.cuda.mem_get_info(gpui)[0]/1024/1024/1024)
      #current_free_mem=min(free_gpus_memorys)
  current_free_mem=min(free_gpus_memorys)
  #==================================#
  for epoch in range(args.num_epochs):
    current_trunk=1
    sampler.set_epoch(epoch)
    #loss_3d=0
    c=0
    N = len(sampler)    
    # current_reconstruct_resolution=args.start_resolution - (args.start_resolution-max(args.end_resolution,2*data_apix))/args.mask_steps * int(epoch/args.mask_epoch)
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

      with torch.cuda.amp.autocast():
        if len(data_unit)==2:
          y,ctf=data_unit

        elif len(data_unit)==3:
          y,ctf,pose=data_unit
          pose=pose.to(device)        

        # ---------get data---------
        y=Variable(y)
        y=y.to(device)

        ctf=ctf.to(device)       

        train_y=y
        trunk_size=y.size(0)
        accum_iter=args.batch_size//trunk_size

        data_block_size=train_y.size(0)
        
        train_y=train_y.view(data_block_size,1,imgsize,imgsize)
        if args.lowpass_input:
          train_y=ift2( torch.mul(ft2(train_y),current_mask))

        # downsample input_image for Encoder .

        if args.downsample_train  :
          downsampled_train_y=downsample_input(train_y,args.downsample_radius)
          #downsampled_train_y=ift2(crop_tensor(ft2(train_y),args.downsample_radius) )
          # downsampled_ctf=crop_tensor(ctf.view(data_block_size,1,imgsize,imgsize),args.downsample_radius)


        # -----calculate noise std -----.
        noise_images=torch.mul(y,noise_mask).view(data_block_size,imgsize,imgsize)
        noise_ft=ft2(noise_images)
        noise_grad=torch.ones(imgsize,imgsize,device=device)
        signal_mu=torch.ones(imgsize,imgsize,device=device)
    
        for mask_i in range(args.mask_steps):
          current_noise_std=1.414*torch.mean (torch.std(noise_ft[new_fourier_masks[mask_i].expand(data_block_size,imgsize,imgsize)>0.49 ]) ,dim=-1)
          noise_grad[new_fourier_masks[mask_i]>0.49]=current_noise_std
          #signal_mu[new_fourier_masks[mask_i]>0.49]=torch.mean(torch.mean(train_y.view(data_block_size,imgsize,imgsize) [new_fourier_masks[mask_i].expand(data_block_size,imgsize,imgsize)>0.49 ]) ,dim=0)



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
        #   downsampled_ctf=image_duplicator(ctf,args.sym_factor)        
        #   downsampled_y=image_duplicator(y,args.sym_factor)          



        #  b = sym_factor * data_block_size   
        b=train_y.size(0)


        # New pre module should downsample particles together

        #print(train_y.size())
        
        train_y=torch.mul(train_y,input_mask)


 
        if args.downsample_train:
          if current_reconstruct_radius>args.downsample_radius  :

            Encoder_Feature_Extractor.eval()
            Encoder_3D_Predictor.eval()       

            features_local=Encoder_Feature_Extractor_Local(train_y) 
            features=Encoder_Feature_Extractor(downsampled_train_y)        
          else:
            #
            if args.random_input:
              downsampled_train_noise=torch.rand_like(downsampled_train_y)
              features=Encoder_Feature_Extractor(downsampled_train_noise)
            else: 
              features=Encoder_Feature_Extractor(downsampled_train_y)  
        else:
          if args.random_noise:
            features=Encoder_Feature_Extractor(train_y+torch.randn_like(train_y)) 
          else:
            features=Encoder_Feature_Extractor(train_y) 

      
        #================3D BLOCK====================

        if mode=='3d' or mode=='both':

          z_3d=Encoder_3D_Predictor(features) 
          if args.downsample_train and current_reconstruct_radius>args.downsample_radius :
            z_3d_local=Encoder_3D_Predictor_Local(features_local)
            z_3d=z_3d+z_3d_local*args.local_refine_scale

          # dim of z_3d should be [batchsize,sym_factor,z_dim]


          rotation_3d=z_3d[:,:6]
          # add random distrubance to escape from  bad initilization in early epochs
          if args.add_distrubance and epoch < args.distrubance_epoch:
            #rotation_3d=rotation_3d + (torch.rand_like(rotation_3d)-0.5)*args.distrubance_ratio
            #using norm distribution instead of uniform distribution.
            distrubance_ratio=args.distrubance_ratio-epoch*(args.distrubance_ratio/ args.distrubance_epoch)
            rotation_3d=rotation_3d + torch.randn_like(rotation_3d)* distrubance_ratio if args.gaussian_distrubance else rotation_3d + (torch.rand_like(rotation_3d)-0.5)*distrubance_ratio
            # rotation_3d=torch.cat([rotation_3d,distrubed_rotation_3d],dim=1)



          if args.latent_code_dim_3d!=0:
            latent_3d=z_3d[:,6:6+args.latent_code_dim_3d]  
          translation_3d=z_3d[:,-2:]*args.dx_scale*imgsize        
          #-----------apply rotation on input coords
          center_plane_grid =  x_coord_plane.to(device)
          #center_plane_grid_triu=torch.triu(center_plane_grid.view(imgsize,imgsize,3))
          center_plane_grid =  center_plane_grid.expand(b,center_plane_grid.size(0),center_plane_grid.size(1))
          rot_3d_matrix=transforms.rotation_6d_to_matrix(rotation_3d)
          rotated_plane_grid=torch.bmm(center_plane_grid,rot_3d_matrix)
          rotated_plane_grid=rotated_plane_grid.view(b,imgsize,imgsize,3)


          y=torch.mul(reconstruct_mask,y)


          #-----------decoder
          if args.reduce_input:
              rotated_plane_grid=rotated_plane_grid[current_mask.expand(b,imgsize,imgsize)>0.49]   
              pred_images_3d=torch.zeros(b,imgsize,imgsize,2,device=device)              

          decoder_output=Decoder_3D(rotated_plane_grid) if args.latent_code_dim_3d==0 else Decoder_3D(rotated_plane_grid,latent_3d)

          if args.reduce_input:
            pred_images_3d[current_mask.expand(b,imgsize,imgsize)>0.49]=decoder_output.view(-1,2)  
          else:
            pred_images_3d=decoder_output



          #-----------apply translation and CTF on pred images
          pred_images_3d = torch.view_as_complex(pred_images_3d)

          pred_images_3d=phase_shift_ft(pred_images_3d,translation_3d,helper_x,helper_y)

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
          y_ft=torch.mul(y_ft,current_mask)

          pred_images_3d=torch.mul(pred_images_3d,current_mask)
          # noise grad could be seen as weighted on different resolution.
          # test no noise_grad
          # y_ft=torch.div(y_ft,noise_grad)
          pred_images_3d=torch.div(pred_images_3d,noise_grad)

          if args.reduce_input:
            y_ft=y_ft[current_mask.expand(b,imgsize,imgsize)>0.49].view(b,-1)
            pred_images_3d=pred_images_3d[current_mask.expand(b,imgsize,imgsize)>0.49].view(b,-1)



          loss_3d= symmetric_complex_loss(pred_images_3d,y_ft,args.sym_factor)



          if args.hard_topk and epoch >= args.topk_start_epoch and epoch < args.topk_end_epoch:
            current_ratio=((args.topk_end_ratio-args.topk_start_ratio)/(args.topk_end_epoch-args.topk_start_epoch))*(epoch-args.topk_start_epoch)+args.topk_start_ratio
            loss_3d=torch.topk(loss_3d,int(data_block_size*current_ratio),largest=False).values  
          elif args.soft_topk:
            loss_3d_mu=loss_3d.mean()
            loss_3d_std=loss_3d.std()
            loss_3d=loss_3d[loss_3d.ge(loss_3d_mu-args.sigma_level*loss_3d_std)*loss_3d.le(loss_3d_mu+args.sigma_level*loss_3d_std)]
            current_ratio=loss_3d.size(0)/data_block_size

          loss_3d=torch.mean(loss_3d)



          loss_3d=torch.div(loss_3d,accum_iter)


          if trunk_size*current_trunk>=args.batch_size:
            update_grad=True
            current_trunk=1
            # loss_3d=torch.div(loss_3d,accum_iter)

          else:
            current_trunk+=1
            update_grad=False



          if mode=='3d' :
            if args.downsample_train :
             
              #if epoch <  args.downsample_train_epoch  :
              if current_reconstruct_radius <  args.downsample_radius  :              
                Encoder_Feature_Extractor_Optimer.zero_grad()
                Encoder_3D_Predictor_Optimer.zero_grad()  
                Decoder_3D_Optimer.zero_grad() 
              else:
                Encoder_Feature_Extractor_Optimer_Local.zero_grad()  
                Encoder_3D_Predictor_Optimer_Local.zero_grad()   
                if epoch >= args.downsample_train_epoch +args.fade_in_epochs :
                  if not args.local_only:
                    Encoder_Feature_Extractor_Optimer.zero_grad()
                    Encoder_3D_Predictor_Optimer.zero_grad()  
                  Decoder_3D_Optimer.zero_grad()                   

            else:
              if epoch <= args.lock_epoch:
                Encoder_Feature_Extractor_Optimer.zero_grad()
              Encoder_3D_Predictor_Optimer.zero_grad()                
              Decoder_3D_Optimer.zero_grad()    


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
                  if epoch  >= args.downsample_train_epoch+ args.fade_in_epochs :
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

            scaler.update()


        c+=data_block_size
        eclipsed_time=time.time()-start_time
        if mode=='2d':
          template = '# [{}/{}] training {:.1%},  Error_2d={:.4f}' 
          line = template.format(epoch, num_epochs, c/N, loss_2d.item())
        elif mode=='3d':
          template = '# [{}/{}] time: {:.1f} s, training {:.1%},  Resolution : {:.1f}, Radius : {:.1f}, Minibatch : {}, Error_3d={:.4f}' 
          line = template.format(epoch, num_epochs, eclipsed_time, c/N, current_reconstruct_resolution , current_reconstruct_radius, current_minibatchsize, loss_3d.item()*accum_iter)             
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
    



    for gpui in range(args.world_size):
      free_gpus_memorys[gpui]=(torch.cuda.mem_get_info(gpui)[0]/1024/1024/1024)
      #current_free_mem=min(free_gpus_memorys)
    current_free_mem=min(free_gpus_memorys)
    # if (epoch+1)%args.save_interval==0 and local_rank==args.write_device : 
    #   with torch.no_grad():
    #     Encoder_Feature_Extractor.eval()
    #     target_y=train_y[0].view(1,2,imgsize,imgsize)
    #     current_particles=target_y.view(2,imgsize,imgsize).detach().cpu().numpy()
    #     DL.write_file(current_particles,args.render_path+'/current_particles_epoch'+str(epoch)+'.mrc')

        # target_layer=Encoder_Feature_Extractor.module.feature_extractor.layers[4][-1].conv2

        # cam_ext=CAM_Extractor(Encoder_Feature_Extractor,target_layer) 
        # feature_map=cam_ext.get_feature_map(train_y[0].view(1,2,imgsize,imgsize))
        # DL.write_file(feature_map,args.render_path+'/features_layer1_epoch'+str(epoch+1)+'.mrc')
         
        # del target_layer,cam_ext,feature_map

        #torch.cuda.empty_cache() 

        # target_layer=Encoder_Feature_Extractor.module.feature_extractor.layers.layer1[-1].conv2
        # cam_ext=CAM_Extractor(Encoder_Feature_Extractor,target_layer) 
        # feature_map=cam_ext.get_feature_map(target_y)

        # # heatmap=cv2.applyColorMap(DL.norm_unsigned(feature_map),cv2.COLORMAP_JET)
        # # cv2.imwrite(args.render_path+'/heatfeature_layer1.png',heatmap)
        # DL.write_file(feature_map,args.render_path+'/features_layer1_epoch'+str(epoch+1)+'.mrc')  
        # del target_layer,cam_ext,feature_map

        # target_layer=Encoder_Feature_Extractor.module.feature_extractor.layers.layer2[-1].conv2
        # cam_ext=CAM_Extractor(Encoder_Feature_Extractor,target_layer) 
        # feature_map=cam_ext.get_feature_map(target_y)
        # DL.write_file(feature_map,args.render_path+'/features_layer2_epoch'+str(epoch+1)+'.mrc') 
        # del target_layer,cam_ext,feature_map

        # target_layer=Encoder_Feature_Extractor.module.feature_extractor.layers.layer3[-1].conv2
        # cam_ext=CAM_Extractor(Encoder_Feature_Extractor,target_layer) 
        # feature_map=cam_ext.get_feature_map(target_y)
        # DL.write_file(feature_map,args.render_path+'/features_layer3_epoch'+str(epoch+1)+'.mrc') 
        # del target_layer,cam_ext,feature_map


        # target_layer=Encoder_Feature_Extractor.module.feature_extractor.layers.layer4[-1].conv2
        # cam_ext=CAM_Extractor(Encoder_Feature_Extractor,target_layer) 
        # feature_map=cam_ext.get_feature_map(target_y)
        # DL.write_file(feature_map,args.render_path+'/features_layer4_epoch'+str(epoch+1)+'.mrc') 
        # del target_layer,cam_ext,feature_map

         
        # del target_layer,cam_ext,feature_map
        #torch.cuda.empty_cache() 
        #Encoder_Feature_Extractor.train()
        #gc.collect()
        #torch.cuda.empty_cache()  

    torch.cuda.empty_cache()          

    if (epoch+1)%args.save_interval==0 and local_rank==args.write_device :   
      

      y=None
      pred_images_3d=None

      with torch.no_grad():
        if latent_code_dim_2d !=0 or args.save_pose:

          test_data_unit=next(iter(train_iterator))            
          if len(test_data_unit)==2:
            y,ctf=test_data_unit

          elif len(test_data_unit)==3:
            y,ctf,pose=test_data_unit
            pose=pose.to(device)          
          y=y.to(device)
          if not args.save_pose:
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
            latent_code_3d=Encoder_3D_Predictor(Encoder_Feature_Extractor(y))[:,6:latent_code_dim_3d+6]
          for slicei in range(imgsize):
            test_pred_images_3D[slicei:slicei+1,...]=Decoder_3D(coords_3d[slicei:slicei+1,...]) if latent_code_3d==None else Decoder_3D(coords_3d[slicei:slicei+1,...],latent_code_3d) 
        


            

          test_pred_images_3D=torch.view_as_complex(test_pred_images_3D)

          if args.save_pose:
            current_loss,index_path=symmetric_complex_loss(test_pred_images_3D,)
            pass

          #test_pred_images_3D=symmetrize_output(test_pred_images_3D)
          # do not use 3dmask to escape from rings.(20250305)
          # but for hard cases this will largely damage output quality.
          if args.lowpass:
            mask_3d=gen_mask(imgsize, (min(int(epoch/args.mask_epoch),mask_steps-1)+1)*0.5/args.mask_steps ).to(device)
            test_pred_images_3D=torch.mul(test_pred_images_3D,mask_3d)

          # generate a new soft here.
          #  def gen_volume_mask(input_tensor,mask_lowpass=20,mask_lowpass_type='butterworth',mask_lowpass_threhold=2,mask_expandsize=3,mask_soft_expand=12):
          #print('generating soft mask...')
          # if epoch+1 >= args.mask_start_epoch:
          #   soft_mask=gen_volume_mask(test_pred_images_3D.real,lowpass_mask_3d,mask_threshold=args.mask_lowpass_threshold,mask_dilation_width=args.mask_dilation_width,mask_padding_width=args.mask_padding_width).to(device) 
          #   DL.write_file(dt(soft_mask),args.render_path+'/rec_3dmask_'+str(epoch+1)+'.mrc')


          test_pred_images_3D=ift3(test_pred_images_3D).real
          test_pred_images_3D=torch.mul(test_pred_images_3D,reconstruct_mask_3d)


      

          test_pred_images_3D=DL.norm_image(dt(test_pred_images_3D))
          
          DL.write_file(test_pred_images_3D,args.render_path+'/rec_3d_'+str(epoch+1)+'_'+str(int(eclipsed_time/60))+'min.mrc')


          test_pred_images_3D=None
          torch.cuda.empty_cache()              
  end_time=time.time()
  time_cost=end_time-start_time
  print("\ntime cost:%.3f s"%time_cost)
       
if __name__ == '__main__':
    main()
