from __future__ import print_function, division
import warnings
warnings.filterwarnings('ignore')
from new_dataloader import StarDataSet

import numpy as np
import pandas as pd
import sys


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

from new_model_utils import *
from utils import *

from untiled import Dataloader as DL

from siren_pytorch import SirenNet


from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

from torch.nn.parallel import  DistributedDataParallel as DDP
from torch.nn.parallel import  DataParallel

from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing as mp

from dataio import *

import time

from fairscale.optim.grad_scaler import ShardedGradScaler

from torch.profiler import profile

def setup_for_distributed(is_master):
  '''
  Beautiful function for disable non-master print
  '''

  import builtins as __builtin__
  builtin_print=__builtin__.print

  def print(*args,**kwargs):
    force=kwargs.pop('force',False)
    if is_master :
      builtin_print(*args,**kwargs)
  __builtin__.print=print


def out_render_2d(y_gen,imgsize,output_name):
  '''
  Used for final 2D render.
  '''

  y_gen=y_gen.view(-1,imgsize*imgsize)
  mrc_array=DL.render_mrc(y_gen.cpu().detach().numpy())

  DL.write_file(mrc_array,output_name)

def out_render(y_gen,imgsize,output_name):
  '''
  Used for final 3D render.
  '''

  y_gen=y_gen.view(imgsize*imgsize*imgsize)
  mrc_array=DL.render_mrc(y_gen.cpu().detach().numpy())

  DL.write_file(mrc_array,output_name)





def main():
  args=get_parser()


  # when using torch.distributed.launch , local_rank shold be set as a argument
  #local_rank=args.local_rank
  # when using torchrun, automaticlly read local_rank
  local_rank=int(os.environ["LOCAL_RANK"])
  #if local_rank==0:
  #if 1:


  # Get a pretrained-decoder

  # I don't want to change Indentation for some reason so I use if 1:
  # with profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_stack=True,record_shapes=True,) as prof:
  if 1:
    # actually sym=2 is the only one succeeded.
    if args.sym:
      assert args.sym_factor==2 or args.sym_factor==4 or args.sym_factor==8, "Only sym factor 2,4,8 supported !"
    # should be False by default for speed up. Maybe needed when handling high resolution information ?
    use_symmetrizer=args.use_symmetrizer


    setup_for_distributed(local_rank==0)
    # standard init
    torch.cuda.set_device(local_rank)
    distr.init_process_group("nccl",world_size=args.world_size,rank=local_rank)

    device=torch.device('cuda')
    imgsize=args.imgsize

    world_size=args.world_size
    n_out=2
    num_epochs = args.num_epochs
    w0=args.w0
    radius=args.input_radius
    w=args.w

    latent_dim=args.z_dim
    T=args.T

    # Creating output directory
    if not os.path.exists(args.save_prefix) and (local_rank==0):
      os.mkdir(args.save_prefix)

    save_path=args.save_prefix
    data_path=args.data_path

    rotation_dim=3
    # fix rotate frozen
    skip_pose=args.skip_read_pose
    if  args.rotate_quater :
        rotation_dim=4
    elif args.rotation6d:
        rotation_dim=6
    elif args.run2d:
        rotation_dim=1


    translation_dim=0
    if not (args.fix_translate or args.skip_shift) :
      translation_dim=2

    ctf_train = None
    ctf_test = None
    full_ctf=None

    ori_imgsize=args.ori_imgsize


    stack_list,index_dict,line_index_dict,total_len,params_dict,full_pose=DL.get_full_info(args.particles_meta,skip_pose,pose_type=args.pose_type)


    full_data_list=DL.get_particles_info(stack_list,index_dict,line_index_dict,params_dict,full_pose=full_pose,skip_pose=skip_pose)

    data_apix=float(params_dict['apix'])

    data_len=total_len
    local_size=data_len/args.world_size

    # full_data=torch.zeros([data_len,imgsize,imgsize])
    data_dtype=np.float16 if args.float16 else np.float32


    if args.load_data_before_train:
      full_data=DL.load_from_particle_stack(stack_list,index_dict,line_index_dict,total_len,prefix=args.data_path,norm=args.norm,dtype=data_dtype)

      #data_len=full_data.shape[0]

      images_train,images_test=DL.split_dataset(full_data,args.split_rate)
      # if args.fast_subset_iter>0:
      #   subset_train,subset_test=DL.split_dataset(full_data,args.fast_subset_rate)


      print('# train:', images_train.shape, file=sys.stderr)
      print('# data type :', images_train.dtype, file=sys.stderr)
      if not (args.standalone or args.use_projections or args.skip_ctf):
        print('#---preparing CTF ...' ,file=sys.stderr)
        # using new data sampling method, so will not carry downsample on CTF independently. 2023/10/23

        # Always full shape CTF.
        # for downsample, ori_image is input size, imgsize is downsampled one
        # for upsample ,ori_image is full_image size ,imgsize is input size

        full_ctf=dt((cuda_get_ctf(params_dict,ori_imgsize,ori_imgsize)))
      ctf_file=args.ctf_file
      if ctf_file !='':
        print('#---preparing CTF ...' ,file=sys.stderr)
        f=open(ctf_file)
        ctf_list=json.load(f)

        dfu=list2tensor(ctf_list['dfu']).to(device)
        dfv=list2tensor(ctf_list['dfv']).to(device)
        dfang=list2tensor(ctf_list['dfang']).to(device)

        ctf_train=create_ctf(imgsize,imgsize,dfu,dfv,dfang,args.down_factor)

        #ctf_train=create_ctf(imgsize,imgsize,dfu,dfv,dfang).float()
      if full_ctf is not None:

        ctf_train,ctf_test=DL.split_dataset(full_ctf,args.split_rate)

        ctf_train = torch.from_numpy(ctf_train).float()
        ctf_test = torch.from_numpy(ctf_test).float()
      if not args.run2d:
        if not args.fix_rotate and not args.fix_translate:
            pose_train=pose_test=None
        else:
            pose_train,pose_test=DL.split_dataset(full_pose,args.split_rate)
            pose_train=torch.from_numpy(pose_train).float()
            pose_test=torch.from_numpy(pose_test).float()
      else:
        pose_train=pose_test=None

      images_train = torch.from_numpy(images_train).half()
      images_test = torch.from_numpy(images_test).float()

      if not args.upsample_infer:

        noise_mask=gen_mask(ori_imgsize,radius,'2d',True).float().to(device)
      else:
        
        noise_mask=gen_mask(imgsize,radius,'2d',True).float().to(device)

      print('# estimating noise Variance using 1000 particles...')
      noise_images=torch.mul(images_train[:1000].to(device),noise_mask)
      noise_ft=ft2(noise_images)
      mean_noise=torch.mean(torch.std(noise_ft,dim=[-2,-1]))

      print('#--NOISE VAR---:'+str( dt(mean_noise)))

      y_train=images_train
      # y_test=images_test

      if pose_train is not None:
        data_train = torch.utils.data.TensorDataset(y_train, ctf_train,pose_train)
      else:
        data_train = torch.utils.data.TensorDataset(y_train, ctf_train) 
    else:
      noise_mask=gen_mask(imgsize,0.45,'2d',True).float().to(device)



      print('skip preload particles')
      print('total : '+ str(total_len) + ' particles')

      if not os.path.exists(args.lmdb):
        if local_rank==0:
          print('making lmdb database. Please re-run command after complished')
          DL.load_and_drop_particles(stack_list,index_dict,line_index_dict,total_len,prefix=data_path,db_path=args.lmdb,dtype=data_dtype)
          print('\n')
          sys.exit()
        else:
          sys.exit()
      else:

        data_train=LmdbDataSet(full_data_list,data_path,db=args.lmdb,imgsize=imgsize,dtype=data_dtype,skip_pose=skip_pose,norm=args.norm,double=args.double)           


      
    train_imgsize=args.train_imgsize

    scale = args.scale


    in_channels=1

    if args.lowpass:
      if args.gaussian:
        gaussian_filters = GaussianPyramid(
                kernel_size=args.kernel_size,
                kernel_variance=args.kernel_variance,
                num_octaves=args.num_octaves,
                octave_scaling=10.,
                include_input=args.include_input
            )

        gaussian_filters=gaussian_filters.to(device)
      else:
        max_radius=int(args.imgsize/2)
        min_radius=6
        grad=int((max_radius-min_radius)/(args.num_octaves-1))
        lowpass_masks=torch.zeros(args.num_octaves,args.imgsize,args.imgsize,device=device)
        for i in range(args.num_octaves):
          lowpass_size=max_radius-i*grad
          lowpass_masks[i]=torch.from_numpy( sigproc.lowpass_mask(args.imgsize, 2, lowpass_size, lowpassorder=2) ).to(device)

      in_channels=args.num_octaves
      if args.include_input:
        in_channels+=1
    if args.run2d:
      x_coord=gen_coords_grid_2d(imgsize,args.rescale)
      coord_dim=2

    else:
      coord_dim=3

      x_coord=gen_coords_grid(train_imgsize,'2d',args.shift_coords)

      x_coord_3d=gen_coords_grid(imgsize,'3d',args.shift_coords)
      x_coord_3d_supers=gen_coords_grid(args.render_size,'3d',args.shift_coords,args.rescale)
      x_coord_3d_ori=gen_coords_grid(ori_imgsize,'3d',args.shift_coords)


    z_dim = args.z_dim


    print('# training with z-dim:', z_dim, file=sys.stderr)


    limit_rate=1.
    if args.limit_rotation:
        limit_rate=0.5
    num_layers = args.decoder_dim
    hidden_dim = args.decoder_layers


    activation=args.activation
    encoder_activation=args.encoder_activation
    last_activation=args.last_activation



    final_dim=latent_dim+rotation_dim+translation_dim

    if args.vae:
      final_dim+=latent_dim
      final_dim+=rotation_dim
      final_dim+=translation_dim
    print('# Encoder final_dim:',final_dim)


    if args.auto_decoder:
      Encoder=Naive_MLP(final_dim,final_dim)
    else:
      Encoder=Build_Encoder(args.encoder_backbone,args.encoder_basename,rotation_dim,translation_dim,in_channels,encoder_activation,last_activation,imgsize,args.norm)

    add_layers=args.add_layers




    Decoder=Fullnet(hidden_dim=args.decoder_dim, num_layers=args.decoder_layers,add_layers=args.add_layers,activation=args.activation, w0=args.w0,w=args.w,n_out=2,in_dim=3,latent_in_dim=args.latent_in_dim,use_symmetrizer=use_symmetrizer)

    Encoder=Encoder.float()
    Decoder=Decoder.float()

    Encoder.apply(weight_init)
    Decoder.apply(weight_init)


    Encoder=Encoder.to(device)

    Decoder=Decoder.to(device)



    minibatch_size = args.minibatch_size


    mask = None

    theta_prior=args.theta_prior
    dx_scale=args.dx_scale

    lr = args.learning_rate
    #"weight_decay":1e-4
    base_optimizer_arguments = { "lr": lr,"eps": 1e-8}
    base_optimizer=torch.optim.Adam


    if args.downsample_train:
      train_y_size=ori_imgsize
    else:
      train_y_size=imgsize

    if args.mask:
        #print('# masking particles', file=sys.stderr)
        # if using downsample_train, this mask is a tiny one
        # all mask should be applied on full image, thus ori_image all the time.
        mask=gen_mask(imgsize,radius)
        # mask = mask.view(imgsize*imgsize,-1)
        input_mask=gen_mask(train_y_size,radius).view(train_y_size,train_y_size)
        reconstruct_mask=gen_mask(train_y_size,args.reconstruct_radius).view(train_y_size,train_y_size)
        mask=mask.to(device)
        input_mask=input_mask.to(device)
        reconstruct_mask=reconstruct_mask.to(device)
    else:
      input_mask=None
      reconstruct_mask=None


    mask_steps=args.mask_steps
    if args.lowpassmask:
      fourier_masks=[]
      new_fourier_masks=[]
      current_reconstruct=[]
      start_res=args.start_res    

      start_radius=1. / (start_res/data_apix) if start_res>0 else 6./imgsize
      print('start fourier radius : %.4f'%start_radius)
      rescale_step=(0.5-start_radius) /(mask_steps-1)
      
      mask_size=train_imgsize if args.downsample_train else imgsize
      for i in range(mask_steps):
        if args.exp_mask:
          rescale_factor=2**(mask_steps-1-i)
          # in older versino here using imgsize/rescale_factor
          # so up to imgsize/2 , 

          full=imgsize if args.butterworth  else 0.5
          fourier_radius=full/rescale_factor
          #print(fourier_radius)
        else:
          fourier_radius=float(i*rescale_step)+start_radius
        if args.butterworth:
          lowpass_mask=torch.from_numpy( DL.lp_mask(imgsize, fourier_radius) ).float().to(device)
        else:
          #print(fourier_radius)
          lowpass_mask=gen_mask(mask_size,fourier_radius).float().to(device)
          #current_reconstrcut

        fourier_masks.append(lowpass_mask)
        new_fourier_masks.append(lowpass_mask)

      for i in range(mask_steps-1):
        new_fourier_masks[i+1]=new_fourier_masks[i+1]-fourier_masks[i]
    #print (torch.ones(train_imgsize,train_imgsize,device=device)[new_fourier_masks[1]>0.5])
    encoder_optim=OSS(params=Encoder.parameters(),optim=base_optimizer,**base_optimizer_arguments)
    decoder_optim=OSS(params=Decoder.parameters(),optim=base_optimizer,**base_optimizer_arguments)

    Encoder=ShardedDDP(Encoder,encoder_optim)
    Decoder=ShardedDDP(Decoder,decoder_optim)

    torch.backends.cudnn.benchmark = True


    
    print ('# start training', file=sys.stderr)


    t1=time.time()
    # if args.render_mask:
    #   render_mask=gen_mask(imgsize,args.render_radius,'3d').to(device).view(1,imgsize,imgsize,imgsize)

    # train_iterator = torch.utils.data.DataLoader(data_train, batch_size=minibatch_size,shuffle=True,pin_memory=True,num_workers=24,drop_last=True)

    sampler=DistributedSampler(data_train)
    train_iterator = torch.utils.data.DataLoader(data_train, batch_size=minibatch_size,shuffle=False,pin_memory=True,num_workers=8,drop_last=True,sampler=sampler)

    #scaler=torch.cuda.amp.GradScaler()
    scaler=ShardedGradScaler()
    helper_x,helper_y=torch.meshgrid(torch.arange(-args.imgsize/2,args.imgsize/2),torch.arange(-args.imgsize/2,args.imgsize/2))
    helper_x=helper_x.to(device)
    helper_y=helper_y.to(device)   
    for epoch in range(num_epochs):
        if epoch < args.decoder_freeze_epoch:
            Decoder.eval()
        else:
            Decoder.train()

        c=0
        b=0
        gen_loss_accum = 0
        kl_loss_accum = 0
        elbo_accum = 0
        sampler.set_epoch(epoch)
        N = len(sampler)
        x=x_coord.to(device)
        if args.lowpassmask:
          mask_index=min(int(epoch/args.mask_epoch),args.mask_steps-1)
          #current_mask=fourier_masks[mask_index]
          current_mask=fourier_masks[mask_index]
          # if int(epoch/args.save_interval)-mask_index>=2:

          #x_mask=current_mask.expand(b,imgsize,imgsize)
          # x=x.view(imgsize,imgsize,3)[current_mask==1]
          #   break
        stepid=0
        for data_unit in train_iterator:


          # if stepid>3:
          #   break
          # stepid+=1
          with torch.cuda.amp.autocast():
            y_var=None

            if len(data_unit)==2:
                if args.decoder_only:
                    y,theta=data_unit
                    theta=theta.to(device)
                    ctf=None
                elif args.skip_ctf:
                    y,pose=data_unit
                    pose=pose.to(device)
                    ctf=None
                else:
                    y,ctf=data_unit
                    pose=None
            elif len(data_unit)==3:
                y,ctf,pose=data_unit
                pose=pose.to(device)
            elif len(data_unit)==4:
                y,ctf,theta,shift=data_unit

                theta=theta.to(device)
                shift=shift.to(device)

            else:
                y=data_unit[0]
                ctf=None
                pose=None

            y=Variable(y)
            y=y.to(device)

            if args.double:
              train_y=y[:,1,:,:] if not args.invert_ed else y[:,0,:,:]
              y=y[:,0,:,:] if not args.invert_ed else y[:,1,:,:]
            else:
              train_y=y

            #b=int(train_y.size(0)/args.sym_factor)
            b=train_y.size(0)
            #print (b)
            orib=b
            


            # train_y_size is input image size (full size in downsample_train case, ori_size in upsample infer case)
            #train_y=train_y.view(b*args.sym_factor,1,train_y_size,train_y_size)
            #y=y.view(b*args.sym_factor,1,args.imgsize,args.imgsize)
            train_y=train_y.view(b,1,train_y_size,train_y_size)
            # if args.downsample_train:
            #   y=torch.nn.AvgPool2d(args.down_factor)(y)
              #y=data_sampling(y,sampling_method=args.sampling_method,down_factor=args.down_factor)


            if not args.load_data_before_train:
              noise_images=torch.mul(train_y,noise_mask).view(b,imgsize,imgsize)
              noise_ft=ft2(noise_images)
              noise_grad=torch.ones(train_imgsize,train_imgsize,device=device)
              if args.lowpassmask and not args.force_full_std:
                if args.downsample_train:
                  noise_ft=noise_ft.view(b,-1,ori_imgsize,ori_imgsize)
                  #y_ft_real=torch.nn.AvgPool2d(args.down_factor)(y_ft.real).view(b,imgsize,imgsize)
                  #y_ft_imag=torch.nn.AvgPool2d(args.down_factor)(y_ft.imag).view(b,imgsize,imgsize)
                  noise_ft_real=data_sampling(noise_ft.real,sampling_method=args.sampling_method,down_factor=args.down_factor).view(b,train_imgsize,train_imgsize)
                  noise_ft_imag=data_sampling(noise_ft.imag,sampling_method=args.sampling_method,down_factor=args.down_factor).view(b,train_imgsize,train_imgsize)

                  noise_ft=torch.stack((noise_ft_real,noise_ft_imag),dim=-1)
                  noise_ft=torch.view_as_complex(noise_ft)

                #noise_ft=torch.mul(noise_ft,current_mask)

                for i in range(mask_steps):
                  #noise_grad[new_fourier_masks[i]>0.5]=
                  #print(noise_ft)
                  #print(noise_ft[new_fourier_masks[i].expand(b,train_imgsize,train_imgsize)>0.5 ])

                  current_noise_std=torch.mean (torch.std(noise_ft[new_fourier_masks[i].expand(b,train_imgsize,train_imgsize)>0.5 ]) ,dim=-1)
                  noise_grad[new_fourier_masks[i]>0.5]=current_noise_std
                  #print(current_noise_std)
                mean_noise=torch.mean(torch.std( noise_ft[current_mask.expand(b,train_imgsize,train_imgsize)>0.49],dim=-1 ))
              else:
                mean_noise=torch.mean(torch.std(noise_ft,dim=[-2,-1]))
        
            if args.sym:
              train_y=image_duplicator(train_y,args.sym_factor,args.rot_manner)
              y=image_duplicator(y,args.sym_factor,args.rot_manner)
              b=train_y.size(0)
            
            #print(torch.min(noise_grad))
            # b=train_y.size(0)
            if args.auto_decoder:
              #z=Encoder(torch.rand_like(y))
              input_latent=torch.ones(b,final_dim).to(device
                )
              #z=Encoder(torch.ones_like(y))
              z=Encoder(input_latent)
            else:
              input_y=train_y
              if not (args.decoder_only or args.freeze_encoder):
                if args.lowpass  :
                  if args.gaussian:
                    input_y=gaussian_filters(input_y)
                  else:
                    input_y=torch_apply_lowpass(input_y,args.num_octaves,lowpass_masks)

                    #input_y=apply_lowpass(input_y,steps=args.num_octaves,include_input=args.include_input)
              if args.mask:
                input_y=torch.mul(input_y,input_mask)                    
              z=Encoder(input_y)
            b=train_y.size(0)

            x=x_coord.to(device)
            x=x.expand(b,x.size(0),x.size(1))
            

            # print(z.size())
            # print(x.size())

            # input_x=x.view(-1,coord_dim)
            # input_x=input_x.expand(b,-1,coord_dim).to(device)

            if not args.freeze_encoder:
              theta=z[:,:rotation_dim]
              z=z[:,rotation_dim:]
              if args.euler:
                if args.sym:
                  theta[...,-1]=theta[...,-1]/2.
                theta=theta*np.pi
                rot_matrix=transforms.euler_angles_to_matrix(theta,'ZYZ')
                x=torch.bmm(x,rot_matrix)
              elif args.af2_quater:
                rot_matrix=af2_quater_matrix(theta)
                x=torch.bmm(x,rot_matrix)
              elif args.rotate_quater:
                quater_input=generate_quater(theta)
                quater_input=quater_input.unsqueeze(1)
                x=transforms.quaternion_apply(quater_input,x)
              elif args.rotation6d:
                if args.theta_to_matrix:
                  rot_matrix=theta_to_matrix(theta)
                else:
                  rot_matrix=transforms.rotation_6d_to_matrix(theta)
                x=torch.bmm(x,rot_matrix)
              elif args.run2d:
                rot = Variable(theta.data.new(b,2,2).zero_())
                theta=theta.view(b)*np.pi

                rot[:,0,0] = torch.cos(theta)
                rot[:,0,1] = -torch.sin(theta)
                rot[:,1,0] = torch.sin(theta)
                rot[:,1,1] = torch.cos(theta)
                x = torch.bmm(x, rot)

            else:

              theta=pose[:,:rotation_dim]
              # print(theta.size())
              pose=pose[:,rotation_dim:]
              rot_matrix=transforms.euler_angles_to_matrix(theta,'ZYZ')
              x=torch.bmm(x,rot_matrix)


            input_x=x.view(b,train_imgsize,train_imgsize,coord_dim)



            if args.reduce_input:
              input_x=input_x[current_mask.expand(b,train_imgsize,train_imgsize)>0.49]

            z_kl=0
            if args.z_dim==0:
              if args.resdecoder:
                y_dec=Decoder(input_channels)
              else:
                if args.reg:
                  if args.ones:
                    zr = Variable(torch.ones(b,1,1,args.reg_dim)).to(device)
                    #zr=torch.ones(b,args.reg_dim)
                    #zr = Variable(x.data.new(b,args.reg_dim).ones())
                  else:
                    zr = Variable(x.data.new(b,1,1,args.reg_dim).normal_())
                  y_dec=Decoder(input_x,zr)
                else:
                  y_dec=Decoder(input_x)
              # input_x=input_x[:,:imgsize*imgsize/2,:]

            else:
              if args.reg:
                zr = Variable(x.data.new(b,z_dim).normal_())
              # using a random noise to generate fvol.
                y_dec=Decoder(input_x,zr)
              else:
                #print (input_x.size())
                #print(z.size())
                y_dec=Decoder(input_x,z)

            # y_mu[current_mask.expand(b,imgsize,imgsize)==1]=y_dec.view(-1,2)

            if args.reduce_input:
              # y_mu=torch.zeros(b,imgsize,imgsize,2).to(device)
              y_mu=torch.zeros(b,train_imgsize,train_imgsize,2,device=device)
              y_mu[current_mask.expand(b,train_imgsize,train_imgsize)>0.49]=y_dec.view(-1,2)
            else:
              y_mu=y_dec.view(b,train_imgsize,train_imgsize,2)

            y_mu=torch.view_as_complex(y_mu)         

            batch_mask=reconstruct_mask
            if not args.skip_shift :
              # current version input shift is not pre-normalized
              #print(z[:,:2])
              if args.freeze_encoder:
                # print('using input shift:')
                # print(shift)
                shift=pose[:,:2]
                if args.swapxy:
                  index=torch.LongTensor([1,0]).to(device)
                  shift=torch.index_select ( shift,1, index )
              else:
                shift=z[:,:2]*dx_scale*imgsize

                z=z[:,2:]
              #print(shift)

              y=y.view(b,train_y_size,train_y_size)
              
              if args.shift_ori:

                batch_mask=shift_masks(imgsize,args.reconstruct_radius,shift)

              # else:

              # else:
              #   batch_mask=reconstruct_mask
              # test all kinds of shift axis.
              # -x,y ; -x,-y; x,-y;
              #
              if args.double_shift:
                shift*=2
              if args.flipx:
                shift[:,0]*=-1
              if args.flipy:
                shift[:,1]*=-1
              y_mu=phase_shift_ft(y_mu,shift,helper_x,helper_y)

            #
            # Apply shift first. * Modified @ Jul 18, 2023.
            #


            if not args.skip_ctf: # apply the CTF filter
              ctf=ctf.to(device)
              if args.sym:
                ctf=image_duplicator(ctf,args.sym_factor)

              # for downsample train ,ctf is ori_imgsize in, but for upsample , need to be imgsize
              # if args.downsample_train:
              #   ctf=ctf.to(device).view(b,-1,ori_imgsize,ori_imgsize)
              # else:
              #   ctf=ctf.to(device).view(b,-1,imgsize,imgsize)

              # CTF always full shape. and downsampled later.
              ctf=ctf.view(b,-1,ori_imgsize,ori_imgsize)

              # Modified 2023/10/23, for new _ sampling _ method
              if args.downsample_train or args.upsample_infer:

                ctf=data_sampling(ctf,sampling_method=args.sampling_method,down_factor=args.down_factor).view(b,train_imgsize,train_imgsize)
                #ctf=torch.nn.AvgPool2d(args.down_factor)(ctf).view(b,imgsize,imgsize)
                #ctf=ctf.to(device).view(b,imgsize,imgsize)
              else:
                ctf=ctf.to(device).view(b,ori_imgsize,ori_imgsize)

              y_mu=torch.mul(y_mu,ctf)


            #
            # Apply reconstruct mask here * Modified @ May28 ,2023 . Previous version modified mask data once shifted.
            #
            if args.mask:
              y=torch.mul(batch_mask,y)

            # if args.fit_noise:
            #   y_mu=y_mu.view(b,imgsize,imgsize)
            #   y_mu=add_gaussian_noise_fourier_space(y_mu,args.SNR)

            kl_div=0
            loss=0
            if args.fourier:
              if not args.upsample_infer:
                y_ft=ft2(y.view(b,ori_imgsize,ori_imgsize))
                if args.downsample_train:
                  y_ft=y_ft.view(b,-1,ori_imgsize,ori_imgsize)
                  #y_ft_real=torch.nn.AvgPool2d(args.down_factor)(y_ft.real).view(b,imgsize,imgsize)
                  #y_ft_imag=torch.nn.AvgPool2d(args.down_factor)(y_ft.imag).view(b,imgsize,imgsize)
                  y_ft_real=data_sampling(y_ft.real,sampling_method=args.sampling_method,down_factor=args.down_factor).view(b,train_imgsize,train_imgsize)
                  y_ft_imag=data_sampling(y_ft.imag,sampling_method=args.sampling_method,down_factor=args.down_factor).view(b,train_imgsize,train_imgsize)

                  y_ft=torch.stack((y_ft_real,y_ft_imag),dim=-1)
                  y_ft=torch.view_as_complex(y_ft)
              else:
                y_ft=ft2(y.view(b,imgsize,imgsize))


              if args.lowpassmask:
                # y_ft_lowpassmask=current_mask.expand(b,imgsize,imgsize)
                # y_ft=y_ft[y_ft_lowpassmask==1].view(b,-1)


                y_ft=torch.mul(y_ft,current_mask)
                y_mu=torch.mul(y_mu,current_mask)
                if not args.load_data_before_train:
                    y_ft=torch.div(y_ft,noise_grad)
                    y_mu=torch.div(y_mu,noise_grad)
                # noise_ft=torch.mul(noise_ft,current_mask)


                if args.reduce_input:
                  y_ft=y_ft[current_mask.expand(b,train_imgsize,train_imgsize)>0.49].view(b,-1)
                  y_mu=y_mu[current_mask.expand(b,train_imgsize,train_imgsize)>0.49].view(b,-1)
                  # noise_ft=noise_ft[current_mask.expand(b,imgsize,imgsize)>0.4].view(b,-1)

              # y_ft=y_ft.view(b,-1,imgsize*imgsize)
              # y_mu=y_mu.view(b,-1,imgsize*imgsize)


              # y_noise=torch.mean(torch.mul(y,noise_mask),dim=0)

              # noise_ft=torch.mul(ft2(y_noise),current_mask)
              # noise_pred=torch.mul(torch.mean( y_mu-y_ft,dim=0).view(imgsize,imgsize),current_mask)


              # noise_score=Complex_L2_Loss(noise_ft,noise_pred)
              if args.sym:
                loss=symmetric_complex_loss(y_mu,y_ft,args.sym_factor)
                if args.hard_topk:
                  loss=torch.topk(loss,int(orib*args.topk_ratio)).values
                loss=torch.mean(loss)

                # noise_loss=symmetric_complex_loss(y_mu,noise_ft,args.sym_factor)
                # loss=loss-noise_loss
              elif args.pixel_loss:
                loss+=((y_mu-y_ft)**2).mean()
              else:
                loss+=complex_loss(y_mu,y_ft)
            # if args.calc_noise:
            #loss=0.5*loss/mean_noise
            default_loss=loss.item()
            # if args.rot_loss:
            #   loss=loss+rot_loss


            if args.real:
              # if args.lowpassmask:
              #   y_mu=torch.mul(y_mu,current_mask)
              y_mu=y_mu.view(-1,imgsize,imgsize)
              if args.lowpassmask:
                y_mu=torch.mul(y_mu,current_mask)
              y_mu=ift2(y_mu)
              y=y.view(-1,imgsize,imgsize)
              y_mu=y_mu.view(-1,imgsize,imgsize)

              if mask is not None:
                y_mu=torch.mul(y_mu,batch_mask)
                y=torch.mul(y,batch_mask)

              y=y.view(-1,imgsize*imgsize)
              y_mu=y_mu.view(-1,imgsize*imgsize)
              # loss=((y_mu-y)**2).mean()
              if args.sym:
                loss+=symmetric_loss(y_mu,y,args.sym_factor)
                if args.L1_loss:
                  loss+=symmetric_image_L1_loss(y_mu,y,args.sym_factor)
              elif args.pixel_loss:
                loss+=((y_mu-y)**2).mean()
              else:
                loss+=torch.sum((y_mu-y)**2,1).mean()

          # if args.encoder_loss:
          #   y_real=ift2(y_mu)
          #   new_z=Encoder(y_real)
          #   old_z=Encoder(input_y)f

          #   loss+=(new_z-old_z)**2

          scaler.scale(loss).backward()
          if not args.freeze_encoder:
            scaler.step(encoder_optim)
          # Added for fullly freeze decoder
          if epoch >=args.decoder_freeze_epoch:
            scaler.step(decoder_optim)

          scaler.update()
          # loss.backward()
          # encoder_optim.step()
          # decoder_optim.step()
          if not args.freeze_encoder:
            encoder_optim.zero_grad()
          if epoch >=args.decoder_freeze_epoch:
            decoder_optim.zero_grad()
          # loss=loss.item()
          #prof.step()

          c += orib


          if args.rot_loss:
            template = '# [{}/{}] training {:.1%},  Error={:.3f}, Rot_L={:.4f}'
            line = template.format(epoch, num_epochs, c/N, loss.item(),rot_loss.item())          
          else:
            template = '# [{}/{}] training {:.1%},  Error={:.3f}'
            line = template.format(epoch, num_epochs, c/N, default_loss)

          print(line,"\r", end="", flush=True)

          # template = '# [{}/{}] training {:.1%},  Error={:.3f}, KL={:.3f}'
          # line = template.format(epoch, num_epochs, c/N,loss,kl_div)
          #break
        #break
        #torch.cuda.empty_cache()
        if (epoch+1)%args.save_interval==0 and local_rank==0:
          y=None
          y_dec=None
          y_mu=None

          with torch.no_grad():

            # 1. coord type , 2d or 3d ?
            if args.run2d:
              x=x_coord.to(device)
              render_size=imgsize
            else:
              if args.downsample_train or args.upsample_infer:
                x=x_coord_3d_ori.to(device)
                render_size=ori_imgsize
                if args.superes:
                  x=x_coord_3d_supers.to(device)
                  render_size=args.render_size
              elif args.superes:
                x=x_coord_3d_supers.to(device)
                render_size=args.render_size
              else:
                x=x_coord_3d.to(device)
                render_size=imgsize


            current_render=0
            if current_render<args.max_render and args.z_dim!=0:
              test_iterator = torch.utils.data.DataLoader(data_test, batch_size=minibatch_size,shuffle=False)

              for data_unit in test_iterator:
                if current_render>=args.max_render:
                  break

                y_var=None

                if len(data_unit)==2:
                    y,ctf=data_unit
                    pose=None
                elif len(data_unit)==3:
                    y,ctf,pose=data_unit
                    pose=pose.to(device)
                else:
                    y=data_unit[0]
                    ctf=None
                    pose=None
                #print(y.size())
                b=y.size(0)
                if args.run2d:
                  x_ex=x.expand(args.max_render,render_size*render_size,2).view(args.max_render,render_size,render_size,2)
                else:
                  x_ex=x.expand(args.max_render,render_size*render_size*render_size,3)
                # if args.positional_encoding:
                #   x_ex=PositionalEncoding(num_encoding_functions=6)
                if args.z_dim!=0:
                    y=Variable(y)
                    y=y.to(device)

                    y=y.view(b,1,imgsize,imgsize)
                    y=y[:args.max_render,:]

                    latent_z_render=Encoder(y)
                    #print(latent_z_render.size())
                    latent_z_render=latent_z_render[:,-latent_dim:]
                    # #print(latent_z.size())
                    # #print(x_ex.size())
                    y_gen_render=Decoder(x_ex,latent_z_render)
                    #print(y_gen.size())
                    y_out=torch.view_as_complex(y_gen_render.view(-1,render_size,render_size,2))
                    y_out=(ift2(y_out)).real
                    # #print(y_out.size())
                    y_render=y_out.cpu().detach().numpy()
                    #print(y_render.shape)
                    DL.write_file(y_render,save_path+'/rec_'+str(epoch+1)+'.mrc')


                current_render=args.max_render

            else:
              x=x.view(render_size,render_size,render_size,3)
              # if args.positional_encoding:
              #   x=PositionalEncoding(num_encoding_functions=6)
              y_gen=torch.zeros(render_size,render_size,render_size,2)
              if args.ones:
                zr = Variable(torch.ones(1,1,1,args.reg_dim)).to(device)
                #zr = torch.ones(1,args.reg_dim)
              elif args.reg:
                zr = Variable(x.data.new(1,1,1,args.reg_dim).normal_())
              for i in range(render_size):
                if args.reg:
                  y_gen[i:i+1,...]=Decoder(x[i:i+1,...],zr).detach()
                else:
                  y_gen[i:i+1,...]=Decoder(x[i:i+1,...]).detach()
                torch.cuda.empty_cache()
              # y_gen=Decoder(x_ex).view(-1,imgsize,imgsize,imgsize,2)
              y_gen=torch.view_as_complex(y_gen)
              if args.lowpass:
                mask_3d=gen_mask(imgsize, (min(int(epoch/args.mask_epoch),args.mask_steps-1)+1)*0.5/args.mask_steps ).float()
                y_gen=torch.mul(y_gen,mask_3d)

              y_gen=ift3(y_gen).real
              
              reconstruct_mask_3d=gen_mask(imgsize,args.reconstruct_radius,'3d').float()
              y_gen=torch.mul(y_gen,reconstruct_mask_3d)


              # if args.render_mask:
              #   y_gen=y_gen*render_mask
              out_render(y_gen,render_size,save_path+'/rec_'+str(epoch+1)+'.mrc')
              y_gen=None
              torch.cuda.empty_cache()
      
    t2=time.time()
    t3=t2-t1
    print("\ntime cost:%.3f s"%t3)
  # if local_rank==0:
  #   prof.export_chrome_trace("trace_long.json")
if __name__ == '__main__':
    main()
