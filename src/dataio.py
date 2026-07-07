from __future__ import print_function, division

import numpy as np
import pandas as pd
import sys

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import torchvision
import scipy.ndimage
from utils import *

import os
import sys
from fourier_utils import *


from untiled import Dataloader as DL



def torch_map_coordinates(input,coordinates,mode='nearest',padding_mode='border'):
  return torch.nn.functional.grid_sample(input, coordinates, mode='bilinear',padding_mode=padding_mode)

from fts import *
import pytorch3d.transforms as transforms

import warnings
warnings.filterwarnings('ignore')


def init_device(d):
  torch.cuda.set_device(d)
  print('# using CUDA device:', d, file=sys.stderr)
  device=torch.device('cuda')

  return device

def slicer(input_volume,coords,mode='real'):
  assert mode=='real' or mode=='fourier', "Unknown mode for "+str(mode)
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

def norm_volume(input_volume,factor=1.):
  max_value=torch.max(input_volume)
  min_value=torch.min(input_volume)

  input_volume=(input_volume-min_value)/(max_value-min_value)
  input_volume=input_volume*factor
  return input_volume

def simulate_data(input_volume=None,b=1,save=False,output_name='test_projs.mrc',ctf=False,shift=False,mode='fourier',mask=None,input_pose=None,coords_shift=True,device=None,angle_mode='euler',factor=1.,norm=False,raw_output=False,input_dfu=None,input_dfv=None,input_dfang=None):
  assert len(input_volume.size())==3 or len(input_volume.size())==4 
  input_volume=input_volume.to(device)
  imgsize=input_volume.size(-1)
  mask_3d=gen_mask(imgsize,0.5,'3d').to(device)
  if norm:
    if len(input_volume.size())==4:
      normlized_volume=torch.zeros(input_volume.size())
      for volumei in range(input_volume.size(0)):
        normlized_volume[volumei,...]=norm_volume(input_volume[volumei],factor)
      input_volume=normlized_volume

    else:
      input_volume=norm_volume(input_volume,factor)
  coords=gen_coords_grid(imgsize,'2d',coords_shift).float()

  # if input_pose!=None:
  #   b=input_theta.size(0)

  input_volume=torch.mul(input_volume,mask_3d)
  coords=coords.expand(b,coords.size(0),coords.size(1))
  #print(coords.size())
  if device is not None:
    coords=coords.to(device)

  if mode=='fourier':
    volume=ft3(input_volume)
    volue=torch.mul(volume,mask_3d)
    if device is not None:
      volume=volume.to(device)
  elif mode=='real':
    volume=input_volume
    #volume=volume.to(device)

  if mask is not None:
    #mask=mask.to(device)
    #volume=volume*mask
    volume=torch.mul(volume,mask)

  #if b>1:
   #[-1,1)
  print('Preparing pose information')
  if input_pose is None:
    if angle_mode=='euler':
      theta=torch.rand((b,3))*360-180
      theta[:,1]=(theta[:,1]+180)/2
      #theta_input=theta_input.unsqueeze(1)
      if device is not None:
        theta=theta.to(device)

      rot_matrix=transforms.euler_angles_to_matrix(theta,'ZYZ').float()
      coords=torch.bmm(coords,rot_matrix) 
    else:
      theta=torch.rand((b,4))*2-1
      quater_input=generate_quater(theta)
      quater_input=quater_input.unsqueeze(1)
      if device is not None:
        quater_input=quater_input.to(device)
      coords=coords.view(b,-1,3)
      coords=transforms.quaternion_apply(quater_input,coords)
  else:
    theta=input_pose[:,:3]
    theta=torch.from_numpy(theta).to(device)
    # theta=torch.zeros((b,3))
    # theta[:,1]=np.pi/2
    #print(theta)
    rot_matrix=transforms.euler_angles_to_matrix(theta,'ZYZ').float()
    #print(coords,rot_matrix)  
    coords=torch.bmm(coords,rot_matrix) 
    # coords=shift_coords(coords, 1., 1., 0., imgsize, imgsize, 1)

  print('slicing...')
  #coords=(coords+1)*imgsize//2
  #coords+=1
  #print(coords)


  if len(volume.size())==3:
    slices=slicer(volume,coords,mode)
  else :
    if mode=='real':
      slices=torch.zeros(b*volume.size(0),imgsize,imgsize)
    else:
      slices=torch.zeros(b*volume.size(0),imgsize,imgsize,dtype=torch.cfloat)
    for volumei in range(volume.size(0)):
      temp=slicer(volume[volumei],coords,mode)

      slices[volumei*b:(volumei+1)*b,...]=temp

  #print(slices.size())
  if ctf:
    if input_dfu is None:
      dfs=torch.randint(5000,25000,(b,2))
      dfu=dfs[...,0]
      dfv=dfs[...,1]
      dfang=torch.rand(b)
    else:
      if device is not None:
        dfu=input_dfu.to(device)
        dfv=input_dfv.to(device)
        dfang=input_dfang.to(device)
      #dfu=torch.from_numpy(input_df['dfu'])
    print('creating ctf...')
    # print(dfu.device)
    ctfs=create_ctf(imgsize,imgsize,dfu,dfv,dfang).view(b,imgsize,imgsize)
    slices=ctfs*slices
  else:
    ctfs=None

  if shift:
    if input_pose==None:
      shifts=(torch.rand(b,2)-0.5)*imgsize/5
    else:
      shifts=input_pose[:,3:]
    print('shifting...') 
    slices=phase_shift_ft(slices,shifts)
  else:
    shifts=None

  
  if mode=='fourier':
    mask_2d=gen_mask(imgsize,0.5,'2d')
    if device is not None:
      mask_2d=mask_2d.to(device)
    slices=torch.mul(slices,mask_2d)
    if not raw_output:
      slices=ift2(slices)

  slices=DL.norm_image(dt(slices))

  if save:
    print('writing...') 
    save_projections=DL.norm_unsigned(slices)
    DL.write_file(save_projections,output_name)
  if ctf and shift:
    return ctfs,slices,theta,shifts
  else:
    return torch.tensor(slices),torch.tensor(theta)




