from __future__ import print_function,division

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data
from siren_pytorch import Siren,Sine
import functools
from torchvision.models import resnet

from torchvision.models import swin_transformer
from torchvision.models import convnext
from torchvision.models import mobilenet,efficientnet


from torchvision.models import MaxVit
from utils import *

from PDO_models import *


def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# Fang K modified @ 20240305.
# First layer dim is not equal so elementwise dot should start from the second layes.
class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None,dropout=0.):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                dropout=dropout
                # Add @ 2024/05/21 . 23:18.  Dropout should be used.

            ))
 
            
            
        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                # x *= rearrange(mod, 'd -> () d')
                # kf@modified

                #x*=mod
                try:
                    #x=torch.mul(x,mod)
                    x*=mod
                except:
                    x*=mod.unsqueeze(1)

        return self.last_layer(x)



class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers,out_dim):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)
            layer_w0 = 48 if is_first else 32
            # is_last = ind == num_layers-1
            # dim_out= out_dim if is_last else dim_hidden
            
            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))
            

            # Add @ 2025/01/04 . 21:47.  try Siren modulator   . Failed. Bad output         
            # self.layers.append(Siren(
            #     dim_in = dim,
            #     dim_out = dim_hidden,
            #     w0 = layer_w0,
            #     use_bias = True,
            #     is_first = is_first,
            #     dropout=True


            # ))
 

        # self.layers.append(nn.Sequential(
        #         nn.Linear(dim, out_dim),
        #         nn.ReLU()))

    def forward(self, z):
        x = z
        #print(z.size())
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z),-1)
        return tuple(hiddens)



def weight_init(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal(m.weight)
        nn.init.constant_(m.bias,0)
    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)    

def weight_init_kaiming(m):
    if isinstance(m,nn.Linear):
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
        nn.init.constant_(m.bias,0)
    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0) 


def name_2_activation(activation_name):
    if activation_name == 'tanh':
        activation = nn.Tanh()
    elif activation_name == 'leakyrelu':
        activation = nn.LeakyReLU()
    # elif activation_name=='SIREN':
    #     activation='SIREN'
    elif activation_name == 'gelu':
        activation = nn.GELU()   
    elif activation_name=='':
        activation=None
    elif activation_name == 'relu':
        activation=nn.ReLU()  
    elif activation_name == 'sigmoid':
        activation=nn.Sigmoid()   
    elif activation_name=='SIREN':
        activation=Sine()

    return activation 


# Given exponential prior on Densities where f(x)=lamda * exp(-lamda*x)
# The fourier space prior should be like F (t) = lamda / ( lamda -i*t)

class Fullnet(nn.Module):
    def __init__(self,hidden_dim=256, num_layers=2,add_layers=1,activation=nn.Tanh
                , w0=40.,w=30.,n_out=1,in_dim=3,latent_code_dim=0,use_symmetrizer=False):
        super(Fullnet, self).__init__() 
        self.num_layers=num_layers
    
        self.net_modulant=Homo(in_dim=in_dim,hidden_dim=hidden_dim,out_dim=n_out,layers_num=num_layers+add_layers,activation=activation,w0=w0,w=w,latent_code_dim=latent_code_dim)

        self.net_envelope=Homo(in_dim=in_dim,hidden_dim=hidden_dim,out_dim=n_out,layers_num=num_layers,activation=activation,w0=w0,w=w,latent_code_dim=latent_code_dim)        

        self.use_symmetrizer=use_symmetrizer
        if use_symmetrizer:
          self.symmetrizer = Symmetrizer()

    def forward(self,x,z=None):
        symx=x
        # if self.use_symmetrizer:
        #  self.symmetrizer.initialize(symx)
        #  symx=self.symmetrizer.symmetrize_input(symx)

        out_mod=self.net_modulant(symx,z)
        out_env=self.net_envelope(symx,z)

        #out_env=torch.clamp(self.net_envelope(symx,z),max=10.)
        #out= torch.exp(out_env) * (1+out_mod)/2.
        # print(out_mod.max())
        # print(out_env.max())
        out= torch.exp(out_env) *out_mod
        #out=out_mod * out_env

        return out


class GaussianPyramid(nn.Module):
    def __init__(self, kernel_size=11, kernel_variance=0.01, num_octaves=4, octave_scaling=10,include_input=False):
        """
        Initialize a set of gaussian filters.

        Parameters
        ---------
        kernel_size: int
        kernel_variance: float
        num_octaves: int
        octave_scaling: int
        """
        super(GaussianPyramid,self).__init__()
        self.kernel_size = kernel_size
        self.variance = kernel_variance
        self.num_dec = num_octaves
        self.scaling = octave_scaling
        self.include_input=include_input
        
        weighting = torch.ones([num_octaves], dtype=torch.float32)
        self.register_buffer('weighting', weighting)
        self.kernels = self.generateGaussianKernels(kernel_size, kernel_variance, num_octaves + 1, octave_scaling)

        self.gaussianPyramid = torch.nn.Conv2d(1, num_octaves + 1,
                                               kernel_size=kernel_size,
                                               padding='same', padding_mode='reflect', bias=False)
        self.gaussianPyramid.weight = torch.nn.Parameter(self.kernels)
        self.gaussianPyramid.weight.requires_grad = False

    def generateGaussianKernels(self, size, var, scales=1, scaling=2):
        """
        Generate a list of gaussian kernels

        Parameters
        ----------
        size: int
        var: float
        scales: int
        scaling: int

        Returns
        -------
        kernels: list of torch.Tensor
        """
        coords = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        xy = torch.stack(torch.meshgrid(coords, coords),dim=0)
        kernels = [torch.exp(-(xy ** 2).sum(0) / (2 * var * scaling ** i)) for i in range(scales)]
        kernels = torch.stack(kernels,dim=0)
        kernels /= kernels.sum((1, 2), keepdims=True)

        kernels = kernels[:, None, ...]
        return kernels

    def forward(self, x):
        out=self.gaussianPyramid(x)
        if self.include_input:
            return out
        else:
            return out[:,1:,...]


class Homo(nn.Module):
    #def __init__(self,hidden_dim=256, num_layers=2,w0=16.,w=16.,out_dim=1,in_dim=3):
  def __init__(self,in_dim=3,hidden_dim=256,out_dim=2,layers_num=2,activation='SIREN',w0=30.,w=20.,latent_code_dim=0):    
    super(Homo, self).__init__()

    self.in_dim=in_dim
    if latent_code_dim==0:
        self.modulator=None
    else:
        #print('Building Modulator with latent_code_dim:',latent_code_dim)
        # force softmax ,using latent code dim as 1 . but not good
        self.modulator=Modulator(dim_in=latent_code_dim,dim_hidden=hidden_dim,num_layers=layers_num,out_dim=out_dim)
        #self.modulator=Modulator(dim_in=latent_code_dim,dim_hidden=hidden_dim,num_layers=layers_num)        
    if activation=='SIREN':
      self.net=SirenNet(dim_in=self.in_dim,dim_hidden=hidden_dim,dim_out=out_dim,num_layers=layers_num,w0_initial=w0,w0=w)
    else:
      self.net=Naive_MLP(in_dim=self.in_dim,hidden_dim=hidden_dim,out_dim=out_dim,hidden_layers=layers_num)
    
    # self.net=Naive_MLP(in_dim=in_dim,hidden_dim=hidden_dim,out_dim=out_dim,hidden_layers=layers_num,activation=activation,w0_initial=w0)
   
  def forward(self,coords,z=None):
    modulate = self.modulator is not None
    # if z is not None:
    #   z=z.expand(coords.size(0),coords.size(1),coords.size(2),coords.size(3),z.size(-1))
   
    mods = self.modulator(z) if modulate else None

    out=self.net(coords,mods)
    return out

# default is [input-> hidden] and [hidden -> output ] layers ,so hidden_layers just add [hidden -> hidden] layers
# No SIREN layer should be called when using naive_MLP 
class Naive_MLP(nn.Module):
  def __init__(self,in_dim,out_dim,hidden_layers=2,hidden_dim=256,activation='tanh',last_activation=None,w0_initial=30.):
    super(Naive_MLP,self).__init__()
    layers=[]
    
    layers.append(nn.Linear(in_dim,hidden_dim))
    activation_layer=name_2_activation(activation)


    layers.append(activation_layer)


    for i in range(hidden_layers):
      layers.append(nn.Linear(hidden_dim,hidden_dim))      
      layers.append(activation_layer)      


    layers.append(nn.Linear(hidden_dim,out_dim))


    # dont use final activation
    if last_activation is not None:
        layers.append(name_2_activation(last_activation)) 

    self.layers=nn.Sequential(*layers)

  def forward(self,x):
    out=self.layers(x)
    return out   



# 


class Feature_Extractor(nn.Module):
  def __init__(self,backbone='resnet',basename='resnet18',in_channels=1,activation='tanh',last_activation='tanh',image_size=64,use_batchnorm=True):
    super(Feature_Extractor,self).__init__()
    assert backbone=='resnet' or backbone=='convnext' or backbone=='pdo' or backbone=='mlp' or backbone=='swin' or backbone=='mobile' or backbone=='efficient' or backbone=='maxvit'
    self.flatten=False
    self.use_batchnorm=use_batchnorm
    if self.use_batchnorm:
        self.batchlayers=torch.nn.BatchNorm2d(in_channels)
    if backbone=='resnet':
        self.feature_extractor=Res2D(basename,in_channels)
        if basename=='layer1':
          self.middle_dim=64
        elif basename=='resnet18' or basename=='resnet34' :
          self.middle_dim=512
        else:
          self.middle_dim=2048   
    elif backbone=='maxvit':
        self.feature_extractor=MaxVIT(image_size,in_channels=in_channels)
        self.middle_dim=512     
    elif backbone=='convnext':
        self.feature_extractor=Convnext2D(basename,in_channels)
        if basename=='tiny' or basename=='small':
          self.middle_dim=768
        elif basename=='basic':
          self.middle_dim=1024  
        elif basename=='large':
          self.middle_dim=1536 
    elif backbone=='pdo':
        self.feature_extractor=PDO_eConvs(in_channels,image_size=image_size)
        self.middle_dim=512   
    elif backbone=='mlp':
        self.flatten=True
        self.flatten_size=in_channels*image_size*image_size
        self.feature_extractor=Naive_MLP(self.flatten_size,256,4,256)         
        self.middle_dim=256 
    elif backbone=='swin':
        if basename=='base':
          self.middle_dim=1024
        else:
          self.middle_dim=768
        self.feature_extractor=Swin(basename,in_channels=in_channels)
    elif backbone=='mobile':
        if basename=='v3_small':
          self.middle_dim=576
        else:
          self.middle_dim=960
        self.feature_extractor=Mobile(basename,in_channels=in_channels)
    elif backbone=='efficient':
        self.middle_dim=1280
        self.feature_extractor=Efficient(basename,in_channels)

    #self.shared_layers=Naive_MLP(self.middle_dim,256,2)



    self.shared_layers=Naive_MLP(self.middle_dim,256,0)



    # self.rotation_layers=Naive_MLP(256,rotation_dim,0,activation=activation,last_activation=last_activation)
    # if shift_dim!=0:
    #   self.shift_layers=Naive_MLP(256,shift_dim,0,activation=activation,last_activation=last_activation)

    # # ADD LATENT IN ENCODER. 20240305 
    # if latent_code_dim!=0:
    #   self.latent_layers=Naive_MLP(256,latent_code_dim,0,activation=activation,last_activation=last_activation)



    # self.shift_dim=shift_dim
    # self.latent_code_dim=latent_code_dim

  def forward(self,x):
    if self.use_batchnorm:
        x=self.batchlayers(x)
    #print(x)
    if self.flatten :
        x=x.view(-1,self.flatten_size)
    
    features=self.feature_extractor(x)
    shared_features=self.shared_layers(features)

    # shared_features=self.feature_extractor(x)

    # rot=self.rotation_layers(shared_features)   
    # out=rot 
    # if self.shift_dim!=0:
    #   shift=self.shift_layers(shared_features)        
    #   out=torch.cat([rot,shift],-1)
    #   if self.latent_code_dim!=0:
    #     latent=self.latent_layers(shared_features)  
    #     out=torch.cat([rot,latent,shift],-1)

    return shared_features




class Encode_Predictor(nn.Module):
  def __init__(self,rotation_dim=6,shift_dim=2,latent_code_dim=0,activation='tanh',last_activation='tanh',hidden_dim=256,add_layers=0):
    super(Encode_Predictor,self).__init__()
    rotation_dims=[hidden_dim]
    shift_dims=[hidden_dim]
    latent_dims=[hidden_dim]
    
    for i in range(add_layers):
      rotation_dims.append(256)
      shift_dims.append(256)
      latent_dims.append(256)

    rotation_dims.append(rotation_dim)
    shift_dims.append(shift_dim)
    latent_dims.append(latent_code_dim)
    #print(rotation_dims)

    self.rotation_layers=Naive_MLP(hidden_dim,rotation_dim,add_layers,activation=activation,last_activation=last_activation) 
    if shift_dim!=0:
      self.shift_layers=Naive_MLP(hidden_dim,shift_dim,add_layers,activation=activation,last_activation=last_activation) 
    if latent_code_dim!=0:
      self.latent_layers=Naive_MLP(hidden_dim,latent_code_dim,add_layers,activation=activation,last_activation=last_activation)  

    self.shift_dim=shift_dim
    self.latent_code_dim=latent_code_dim    
  def forward(self,x):
    rot=self.rotation_layers(x)   
    out=rot 
    if self.shift_dim!=0:
      shift=self.shift_layers(x)        
      out=torch.cat([rot,shift],-1)
      if self.latent_code_dim!=0:
        latent=self.latent_layers(x)  
        out=torch.cat([rot,latent,shift],-1)    
    return out


class MaxVIT(nn.Module):
    def __init__(self,input_size,in_channels=1,stem_channels=64):
        super(MaxVIT,self).__init__()
        base_model=MaxVit([input_size,input_size],stem_channels=stem_channels,block_channels=[64, 128, 256, 512],block_layers=[2, 2, 5, 2],head_dim=32,stochastic_depth_prob=0.2,partition_size=2,)
        base_model.stem[0][0]=nn.Conv2d(in_channels, stem_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        base_model.classifier[5]=nn.Identity()
        self.layers=base_model

    def forward(self,x):
        return self.layers(x)        


class Efficient(nn.Module):
    def __init__(self,basename='v2_s',in_channels=1):
        super(Efficient,self).__init__()
        base_model=efficientnet.efficientnet_v2_s()
        base_model.features[0][0]=nn.Conv2d(in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        base_model.classifier=nn.Identity()
        self.layers=base_model
    def forward(self,x):
        return self.layers(x)


class Swin(nn.Module):
    def __init__(self,basename='v2_t',in_channels=1):
      super(Swin,self).__init__()
      base_model=swin_transformer.swin_v2_t()
      base_model.features[0][0]=nn.Conv2d(in_channels ,96, kernel_size=(4, 4), stride=(4, 4))     
      base_model.head=nn.Identity()
      self.layers=base_model
    def forward(self,x):
      return self.layers(x)


class Mobile(nn.Module):
  def __init__(self,basename='v3_small',in_channels=1):
    super(Mobile,self).__init__()
    if basename=='v3_small':
      base_model=mobilenet.mobilenet_v3_small()
    else:
      base_model=mobilenet.mobilenet_v3_large()      
    # 1.
    base_model.features[0][0]=nn.Conv2d(in_channels,16,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)

    # 2. 
    base_model.classifier=nn.Identity()
    self.layers=base_model


  def forward(self,x):
    return self.layers(x)    


class ResDecoder(nn.Module):
    def __init__(self):
        super(ResDecoder,self).__init__()
        base_res=resnet.resnet18()
        base_res.fc=nn.Linear(512,2)
        self.layers=base_res
    def forward(self,x):
        return self.layers(x)


class Res2D(nn.Module):
  def __init__(self, basename='resnet18',in_channels=1):
    super(Res2D,self).__init__()
    #self.input_norm=nn.BatchNorm2d(in_channels)
    if basename=='resnet18' or basename=='layer1':
      base_res= resnet.resnet18()
    elif basename=='resnet34':
      base_res= resnet.resnet34()  
    elif basename=='resnet50':
      base_res= resnet.resnet50()  
    elif basename=='resnet101':
      base_res= resnet.resnet101() 
    elif basename=='resnet152':
      base_res= resnet.resnet152()       
    #1. replace input channel
    base_res.conv1=nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3,bias=False)
    #2. replace output layer
    base_res.fc=nn.Identity()

    # special layer1:
    if basename=='layer1':
      layers=[]
      layers.append(base_res.conv1)
      layers.append(base_res.bn1)
      layers.append(base_res.relu)
      layers.append(base_res.maxpool)
      layers.append(base_res.layer1)
      layers.append(base_res.avgpool)
      layers.append(nn.Flatten(1))

      special_base_res=nn.Sequential(*layers)

      self.layers=special_base_res
    else:
      self.layers=base_res

  def forward(self,x):
    #x=self.input_norm(x)
    return self.layers(x)
     

class Convnext2D(nn.Module):
  def __init__(self,basename='tiny',in_channels=1):
    super(Convnext2D,self).__init__()
    if basename=='tiny':
      base_model=convnext.convnext_tiny()
    if basename=='small':
      base_model=convnext.convnext_small()
    if basename=='basic':
      base_model=convnext.convnext_base()      

    #1. replace  input channel
    base_model.features[0][0]=nn.Conv2d(in_channels,96,kernel_size=(4,4),stride=(4,4))
    #2. replace output layer
    base_model.classifier[2]=nn.Identity()

    self.layers=base_model

  def forward(self,x):
    return self.layers(x)

