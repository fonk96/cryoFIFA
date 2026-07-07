import torch
import numpy as np
from untiled import Dataloader as DL
import os
from typing import Optional
from utils import *
from fourier_utils import *
from new_dataloader import LmdbDataSet

from merged_models import *

from scipy.ndimage import binary_fill_holes
import scipy.ndimage.morphology as spmorph

import sigproc
import fourier

def torch_map_coordinates(input_tensor,coordinates,mode='nearest',padding_mode='border'):
  return torch.nn.functional.grid_sample(input_tensor, coordinates, mode='bilinear',padding_mode=padding_mode)


def slicer(input_volume,coords,mode='fourier'):
  assert mode=='real' or mode=='fourier', "Unknown mode for "+str(mode)
  #print(input_volume.size())
  #print(coords.size())
  b=coords.size(0)
  coords=coords.view(b,1,1,-1,3) 
  imgsize=input_volume.size(-1)

  if mode=='real':
    slices=torch_map_coordinates(input_volume.expand(b,1,imgsize,imgsize,imgsize),coords).view(b,imgsize,imgsize)
  else:
    real_part=input_volume.real
    imag_part=input_volume.imag
    slices_real=torch_map_coordinates(real_part.expand(b,1,imgsize,imgsize,imgsize),coords)
    slices_imag=torch_map_coordinates(imag_part.expand(b,1,imgsize,imgsize,imgsize),coords)

    slices=torch.stack((slices_real.view(b,imgsize,imgsize,1),slices_imag.view(b,imgsize,imgsize,1)),dim=-1)
    slices=torch.view_as_complex(slices).view(b,imgsize,imgsize)


  return slices



def data_constructor(data_path,particles_meta,imgsize,lmdb_path='',load_data_before_train=False,pose_type=3,skip_pose=True,norm=False,data_dtype=np.float16,split_rate=1,skip_ctf=False,fix_rotate=False,fix_translate=False,upsample_infer=False,downsample_train=False,downsample_imgsize=48,double=False,with_name=False):
    #print(data_path,particles_meta,imgsize,lmdb_path)
    stack_list,index_dict,line_index_dict,total_len,params_dict,full_pose=DL.get_full_info(particles_meta,skip_pose,pose_type=pose_type)
    full_data_list=DL.get_particles_info(stack_list,index_dict,line_index_dict,params_dict,full_pose=full_pose,skip_pose=skip_pose)

    assert os.path.exists(lmdb_path),"Cant find lmdb path for "+str(lmdb_path)


    print('skip preload particles')
    print('total : '+ str(total_len) + ' particles')
    #used_data=full_data_list[:int(total_len*split_rate)]
    data_train=LmdbDataSet(full_data_list,data_path,db=lmdb_path,imgsize=imgsize,dtype=data_dtype,skip_pose=skip_pose,norm=norm,double=double,with_name=with_name)
    
    return data_train,total_len,params_dict['apix']

def model_constructor(name='Feature_Extrator',backbone='resnet',basename='resnet18',in_channels=1,activation='tanh',last_activation='tanh',image_size=64,use_batchnorm=False,rotation_dim=6,shift_dim=2,latent_code_dim=0,hidden_dim=256,num_layers=2,add_layers=1,out_dim=2,in_dim=3,use_symmetrizer=False,w0=48,w=32):

  '''
   model dict : Encoder_Feature_Extractor,
                Encoder_2D_Predictor,
                Encoder_3D_Predictor,
                Decoder_2D,
                Decoder_3D
  '''
  assert name=='Feature_Extrator' or name=='Encode_Predictor' or name=='Decoder'


  if name=='Feature_Extrator':
    model=Feature_Extractor(backbone=backbone,basename=basename,in_channels=in_channels,activation=activation,last_activation=last_activation,image_size=image_size,use_batchnorm=use_batchnorm)
  elif name=='Encode_Predictor':
    model=Encode_Predictor(rotation_dim=rotation_dim,shift_dim=shift_dim,latent_code_dim=latent_code_dim,activation=activation,last_activation=last_activation,hidden_dim=hidden_dim,add_layers=add_layers)
  elif name=='Decoder':
    #print('Building Decoder with latent_code_dim:',latent_code_dim)
    model=Fullnet(hidden_dim=hidden_dim, num_layers=num_layers,add_layers=add_layers,activation='SIREN'
                , w0=w0,w=w,n_out=out_dim,in_dim=in_dim,latent_code_dim=latent_code_dim,use_symmetrizer=use_symmetrizer)

  return model


# must use new preprocesser for downsample_train.
def preprocesser(lowpass=True,gaussian=False,preprocessing_output_channels=2,imgsize=128,kernel_size=11,kernel_variance=0.01,include_input=False,exp_mask=False,min_radius=6):
  pre_module=None
  if lowpass:
    if gaussian:
      gaussian_filters = GaussianPyramid(
                kernel_size=kernel_size,
                kernel_variance=kernel_variance,
                num_octaves=preprocessing_output_channels,
                octave_scaling=10.,
                include_input=include_input
      )

      pre_module=gaussian_filters
    else:
      max_radius=int(imgsize/2)
      min_radius=min_radius
      grad= 0 if preprocessing_output_channels ==1 else int((max_radius-min_radius)/(preprocessing_output_channels-1))
      pre_module=torch.zeros(preprocessing_output_channels,imgsize,imgsize)
      res_max_radius=min_radius + (preprocessing_output_channels-1)*grad
      for i in range(preprocessing_output_channels):
        lowpass_size=res_max_radius-i*grad

        #change lowpassorder = 6, @20241011

        pre_module[i]=torch.from_numpy( sigproc.lowpass_mask(imgsize, 2, lowpass_size, lowpassorder=6) )

    # preprocessed_channels=preprocessing_output_channels
    # if include_input:
    #   preprocessed_channels+=1

  return pre_module
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



def which_half_space(coords, eps=1e-6):
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    slab_xyz = (x < -eps)
    slab_yz = torch.logical_and(torch.logical_and(x > -eps, x < eps), y < -eps)
    slab_z = torch.logical_and(torch.logical_and(torch.logical_and(x > -eps, x < eps),
                                                 torch.logical_and(y > -eps, y < eps)), z < -eps)

    return torch.logical_or(slab_xyz, torch.logical_or(slab_yz, slab_z))


def symmetrize_output(input_tensor):
  ''' 
  All input has shape of [b,-1,N,N] where N is even number and DC component located at [N//2,N//2] 
  what's more insteresting is that using fourier_mask (also centered at N//2), all input are perfectlly located in a circluar 
     which center is [N//2,...] and [:,:,0] is 0 by default
     So I just need to place all negative component exactly the same value as positive ones

  '''
  output_tensor=input_tensor.clone().detach()
  center=output_tensor.size(-1)//2
  output_tensor[1:center,1:center,1:center].real=torch.flip(output_tensor[center+1:,center+1:,center+1:],dims=[0,1,2]).real
  output_tensor[1:center,1:center,1:center].imag=-torch.flip(output_tensor[center+1:,center+1:,center+1:],dims=[0,1,2]).imag

  return output_tensor

def gen_lowpassmask_3d(imgsize,radwn,mask_lowpass_order=6):
  
  return sigproc.lowpass_mask(imgsize,3,radwn,lowpassorder=mask_lowpass_order)



def generate_tight_mask(x, thresh_factor, near, far, type='cosine', use_abs=False, color=None, use_relative_thresh=True):
    """Return a tight mask from the input (which is thresholded). 
    Threshold value is thresh_factor * n.percentile(x, 99.99) if use_relative_thresh, otherwise direct threshold
    The mask is 1.0 until a distance of 'near' from the closest True value, and
    becomes softly 0.0 at a distance of 'far'. The softness type can be either
    'linear' or 'cosine'
    Near and Far should be in units of voxels. """
    D = len(x.shape)
    N = x.shape[0]
    if color is not None:
        X = fourier.fft(x)
        X *= rotational_expand(color, N, D)
        x = fourier.ifft(X)
    if use_relative_thresh:
        thresh = thresh_factor * np.percentile(x.ravel(), 99.99)
    else:
        thresh = thresh_factor
    dist = spmorph.distance_transform_edt((np.abs(x) if use_abs else x) <= thresh)
    clipdist = np.clip((dist - near)/(far - near), 0.0, 1.0)
    if type == 'linear':
        return 1.0 - clipdist
    if type == 'cosine':
        return 0.5+0.5*np.cos(clipdist*np.pi)



def gen_volume_mask(input_tensor,lowpass_mask_3d,mask_threshold=128,mask_dilation_width=3,mask_padding_width=12):
  '''
    Generate a soft mask for input volume.
    The refine mask will be applied on input_images using predicted pose information:-->get projection slice ; and translation information :--> ; and then only masked image signal are used for loss computation and backward.

    input volume should be fourier 
  '''
  # 0. lowpass
  # 1. binarization
  # 2. dilation
  
  # sigproc: def lowpass_mask(N, d, radwn=None, lowpasstype='butterworth', lowpassorder=1):

  imgsize=input_tensor.size(-1)

  lowpassed_volume=ift3(torch.mul(input_tensor,lowpass_mask_3d)).real
  #DL.write_file(dt(lowpassed_volume),'rec_3d_lp_test.mrc')
  true_threshold=mask_threshold*(torch.max(lowpassed_volume)-torch.min(lowpassed_volume))+torch.torch.min(lowpassed_volume)
  binary_volume=dt(lowpassed_volume>true_threshold).astype(np.int32)
  
  binary_mask=binary_fill_holes(binary_volume).astype(np.int32)

  binary_mask=generate_tight_mask(binary_mask.astype(np.float32), 0.1, mask_dilation_width, mask_dilation_width+mask_padding_width)
  
  return torch.from_numpy(binary_mask>0.1).float()

def gen_mask_projection(input_mask_tensor,rotated_plane_grid,input_shift,helper_x,helper_y):
  '''
    Return rotated and shifted mask projections.
    input_mask_tensor : [imgsize,imgsize,imgsize] , should be positive values . should be real space .
    rotated_plane_grid: [ b, -1 ,3]
    input_shift : [b,2]

    --------

    return : [b, imgsize, imgisze]
  '''
  
  imgsize=input_mask_tensor.size(-1)
  b=input_shift.size(0)
  input_mask=ft3(input_mask_tensor)
  projections=slicer(input_mask,rotated_plane_grid.float())
  #shifted_projections=ift2(projections).real
  shifted_projections=ift2(phase_shift_ft(projections,input_shift.float(),helper_x,helper_y)).real

  #binary_mask=(shifted_projections>0).float()
  binary_mask=(shifted_projections>0.95).float()
  return binary_mask


def downsample_input(input_tensor,output_radius,soft_edge=0.0):
  '''
    Input size should be [ b ,1 ,imgsize ,imgsize ]
    Output size will be [ b, 1 ,croped_size , croped_size ]

    soft_edge (fraction of the crop half-width): if >0, the cropped Fourier box is
    apodized with a raised-cosine circular window that tapers to 0 over the outer
    `soft_edge` fraction, instead of a hard box crop. A hard box in Fourier space
    convolves the downsampled real image with a sinc (ringing); the soft window
    removes it. soft_edge=0 reproduces the original hard crop exactly.
  '''
  b=input_tensor.size(0)
  channel=input_tensor.size(1)
  ori_size=input_tensor.size(-1)
  imgcenter=ori_size//2

  new_tensor=torch.zeros(b,channel,2*output_radius,2*output_radius,device=input_tensor.device)
  input_ft=ft2(input_tensor)

  ft_slice=input_ft[:,:,imgcenter-output_radius:imgcenter+output_radius,imgcenter-output_radius:imgcenter+output_radius]

  if soft_edge and soft_edge > 0:
    d=2*output_radius
    cc=d//2
    yy,xx=torch.meshgrid(torch.arange(d,device=input_tensor.device),
                         torch.arange(d,device=input_tensor.device))
    rr=torch.sqrt((xx-cc).float()**2+(yy-cc).float()**2)/(d/2.0)  # 0 at DC, 1 at box edge
    t=torch.clamp((rr-(1.0-soft_edge))/soft_edge,0.0,1.0)
    window=0.5*(1.0+torch.cos(3.141592653589793*t))              # 1 in the interior -> 0 at the edge
    ft_slice=ft_slice*window

  try:
    new_tensor[...]=ift2(ft_slice).real
  except:
    print(ft_slice.size())
    print(input_ft.size())
  return new_tensor



def crop_tensor(input_tensor,output_radius):
  '''
    Input size should be [...,imgsize ,imgsize ]
    Output size will be [ ...,croped_size , croped_size ]
  '''

  ori_size=input_tensor.size(-1)
  imgcenter=ori_size//2
  new_tensor=torch.zeros(*input_tensor.shape[:-2],output_radius*2,output_radius*2,device=input_tensor.device)
  #new_tensor=torch.zeros(b,channel,2*output_radius,2*output_radius,device=input_tensor.device)

  new_tensor[...]=input_tensor[...,imgcenter-output_radius:imgcenter+output_radius,imgcenter-output_radius:imgcenter+output_radius]

  return new_tensor

def cosine_decay(epoch, T, start, end):
    t = min(max(epoch / max(1, T), 0.0), 1.0)
    # end + 0.5*(start-end)*(1+cos(pi*t))
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))

def make_shell_weights_midband(mask_steps, base=1.0, gain=1.0, peak=0.45, width=0.18, device="cuda", dtype=torch.float32):
    """
    base: 低频地板（建议 >=1）
    gain: 中频提升幅度（注意你是乘到两边FFT上，loss相当于乘 w^2；gain别太大）
    peak: 中频峰值位置（0~1，按 shell index 归一化）
    width: 峰的宽度（0~1）
    """
    i = torch.arange(mask_steps, device=device, dtype=dtype)
    t = i / max(1, mask_steps - 1)  # [0,1]
    bump = torch.exp(-0.5 * ((t - peak) / max(width, 1e-6)) ** 2)
    w = base + gain * bump
    return w

# import torch

def uniform_s2_moment_loss(
    R: torch.Tensor,
    axis: Optional[int] = 2,
    eps: float = 1e-8,
    use_all_axes: bool = True,
    cross_axes: bool = True,
) -> torch.Tensor:
    """
    Improved S^2 uniformity regularizer for rotations.

    What’s improved vs your original:
    1) Optionally applies the S^2 moment constraints to ALL three axes (columns) instead of only one.
       This discourages "uniform z-axis but collapsed roll" type failures.
    2) Optional cross-axis decorrelation: E[u_i u_j^T] ≈ 0 for i!=j (for columns of R),
       which further discourages degenerate distributions.

    Args:
      R: (B,3,3)
      axis: which column to use if use_all_axes=False. default 2
      eps: numerical stability
      use_all_axes: if True, sum losses over axes {0,1,2}
      cross_axes: if True, add cross-axis decorrelation penalty

    Returns:
      scalar loss
    """
    assert R.dim() == 3 and R.shape[-2:] == (3, 3)
    I = torch.eye(3, device=R.device, dtype=R.dtype)

    axes = (0, 1, 2) if use_all_axes else (int(axis),)
    loss = R.new_tensor(0.0)

    # per-axis S^2 moment matching: E[u]=0 and E[uu^T]=I/3
    for a in axes:
        u = R[:, :, a]  # (B,3)
        m1 = u.mean(dim=0)                       # (3,)
        M2 = (u.unsqueeze(2) @ u.unsqueeze(1)).mean(dim=0)  # (3,3)

        loss_m1 = (m1 * m1).sum()
        loss_m2 = ((M2 - I / 3.0) ** 2).sum()
        loss = loss + loss_m1 + loss_m2

    # cross-axis decorrelation: E[u_i u_j^T] ≈ 0
    # (useful when you constrain all axes; cheap and helps avoid partial SO(3) collapse)
    if cross_axes and use_all_axes:
        for i in (0, 1, 2):
            for j in (0, 1, 2):
                if i == j:
                    continue
                ui = R[:, :, i]  # (B,3)
                uj = R[:, :, j]  # (B,3)
                Cij = (ui.unsqueeze(2) @ uj.unsqueeze(1)).mean(dim=0)  # (3,3)
                loss = loss + (Cij * Cij).sum()

    return loss


# def uniform_s2_moment_loss(R: torch.Tensor, axis: int = 2, eps: float = 1e-8) -> torch.Tensor:
#     """
#     R: (B,3,3)
#     axis: which column vector to use as direction on S^2 (0/1/2). default 2 (z-axis)
#     """
#     u = R[:, :, axis]  # (B,3)
#     # 1st moment
#     m1 = u.mean(dim=0)                 # (3,)
#     loss_m1 = (m1 * m1).sum()

#     # 2nd moment
#     M2 = (u.unsqueeze(2) @ u.unsqueeze(1)).mean(dim=0)  # (3,3)
#     I = torch.eye(3, device=R.device, dtype=R.dtype)
#     loss_m2 = ((M2 - I/3.0) ** 2).sum()

#     return loss_m1 + loss_m2


def reconstruct_fourier_masks(start_resolution=30,data_apix=1.,imgsize=128,downsample_train=False,mask_steps=32,butterworth=False,exp_mask=False,device=None,end_resolution=6,triu=False):
  fourier_masks=torch.zeros(mask_steps,imgsize,imgsize,device=device)
  new_fourier_masks=torch.zeros(mask_steps,imgsize,imgsize,device=device)
  start_res=start_resolution  
  end_res=max(end_resolution,2*data_apix)


  start_radius=1. / (start_res/data_apix) if start_res>0 else 6./imgsize
  end_radius  = 1. / (end_res/data_apix)
  #print('start fourier radius : %.4f'%start_radius)
  rescale_step=(end_radius-start_radius) /(mask_steps-1)
      
  mask_size=imgsize
  for i in range(mask_steps):
    if exp_mask:
      rescale_factor=2**(mask_steps-1-i)
      # in older versino here using imgsize/rescale_factor
      # so up to imgsize/2 , 
      # if steps=3, factor= 2**[2,1,0]

      full=imgsize if butterworth  else 0.5
      fourier_radius=start_radius+(full-start_radius)/rescale_factor
    else:
      fourier_radius=float(i*rescale_step)+start_radius
      #print(fourier_radius)
    if butterworth:
      lowpass_mask=torch.from_numpy( DL.lp_mask(imgsize, fourier_radius) ).float()
    else:
      lowpass_mask=gen_mask(mask_size,fourier_radius).float()
    # Ignore DC component
    lowpass_mask[imgsize//2,imgsize//2]=0

    new_fourier_masks[i]=lowpass_mask.to(device)
    if triu:
      lowpass_mask=torch.triu(lowpass_mask)
    fourier_masks[i]=lowpass_mask.to(device)


  for i in range(mask_steps-1):
    new_fourier_masks[i+1]=(new_fourier_masks[i+1]-fourier_masks[i])



  return fourier_masks,new_fourier_masks

