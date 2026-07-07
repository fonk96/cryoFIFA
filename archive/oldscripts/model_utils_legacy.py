from __future__ import print_function,division

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data
from siren_pytorch import Siren
import functools
from torchvision.models import resnet
from siren_pytorch import SirenWrapper
from torchvision.models import swin_transformer
from torchvision.models import convnext
from torchvision.models import mobilenet,efficientnet
from siren_pytorch import SirenNet
from torchvision.models import MaxVit
from utils import *

from PDO_models import *

def weight_init(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal(m.weight)
    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)

def weight_init_kaiming(m):
    if isinstance(m,nn.Linear):
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)

def name_2_activation(activation_name):
    if activation_name == 'tanh':
        activation = nn.Tanh()
    elif activation_name == 'leakyrelu':
        activation = nn.LeakyReLU()
    elif activation_name=='SIREN':
        activation='SIREN'
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
                , w0=40.,w=30.,n_out=1,in_dim=3,use_symmetrizer=False,pe=False,num_encoding=6):
        super(Fullnet, self).__init__() 
        self.num_layers=num_layers
        #self.net_modulant=Homo(hidden_dim,self.num_layers+add_layers,activation,w0,w,n_out,in_dim)
        #self.net_envelope=Homo(hidden_dim,num_layers,activation,w0,w,n_out,in_dim)

        self.net_modulant=Homo(in_dim=in_dim,hidden_dim=hidden_dim,out_dim=n_out,layers_num=num_layers+add_layers,activation=activation,w0=w0,w=w,pe=pe,num_encoding=num_encoding)
        self.net_envelope=Homo(in_dim=in_dim,hidden_dim=hidden_dim,out_dim=n_out,layers_num=num_layers,activation=activation,w0=w,w=w,pe=pe,num_encoding=num_encoding)        
        self.use_symmetrizer=use_symmetrizer
        if use_symmetrizer:
          self.symmetrizer = Symmetrizer()

    def forward(self,x):
        symx=x
        if self.use_symmetrizer:
          self.symmetrizer.initialize(symx)         
          symx=self.symmetrizer.symmetrize_input(symx)
        
        out_mod=self.net_modulant(symx)
        out_env=self.net_envelope(symx)
        out=torch.exp(out_env) * out_mod
        if self.use_symmetrizer:
          out=self.symmetrizer.antisymmetrize_output(out)
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



class FullHetero(nn.Module):
    def __init__(self,latent_in_dim=4,latent_out_dim=256,coord_net_in_dim=3,coord_net_out_dim=256,hidden_dim=256,out_dim=2,layers_num=2,w0=30.,w=1.,use_symmetrizer=False,add_layers=2,pe=False,num_encoding=6):
        super(FullHetero, self).__init__() 
        #self.net_modulant=Homo(hidden_dim,self.num_layers+add_layers,activation,w0,w,n_out,in_dim)
        #self.net_envelope=Homo(hidden_dim,num_layers,activation,w0,w,n_out,in_dim)

        self.net_modulant=Hetero(latent_in_dim=latent_in_dim,latent_out_dim=latent_out_dim,coord_net_in_dim=coord_net_in_dim,coord_net_out_dim=coord_net_out_dim,hidden_dim=hidden_dim,out_dim=out_dim,layers_num=layers_num+add_layers,w0=w0,w=w,pe=False,num_encoding=6)  
        self.net_envelope=Hetero(latent_in_dim=latent_in_dim,latent_out_dim=latent_out_dim,coord_net_in_dim=coord_net_in_dim,coord_net_out_dim=coord_net_out_dim,hidden_dim=hidden_dim,out_dim=out_dim,layers_num=layers_num,w0=w0,w=w,pe=False,num_encoding=6) 


        self.use_symmetrizer=use_symmetrizer
        if use_symmetrizer:
          self.symmetrizer = Symmetrizer()

    def forward(self,x,z):
        symx=x
        if self.use_symmetrizer:
          self.symmetrizer.initialize(symx)
          symx=self.symmetrizer.symmetrize_input(symx)
        
        out_mod=self.net_modulant(symx,z)
        out_env=self.net_envelope(symx,z)
        out=torch.exp(out_env) * out_mod
        if self.use_symmetrizer:
          out=self.symmetrizer.antisymmetrize_output(out)
        return out


# class FullHetero(nn.Module):
#     def __init__(self,z_dim,hidden_dim=256, num_layers=2,add_layers=1,activation='tanh'
#                 , w0=40.,w=30.,n_out=1,in_dim=3):
#         super(FullHetero, self).__init__() 
#         self.num_layers=num_layers
#         self.net_modulant=Hetero(z_dim,hidden_dim,self.num_layers+add_layers,name_2_activation(activation),w0,w,n_out,in_dim)
#         self.net_envelope=Hetero(z_dim,hidden_dim,num_layers,activation,w0,w,n_out,in_dim)

#         # self.symmetrizer = Symmetrizer()

#     def forward(self,x,z):
#         symx=x

#         # self.symmetrizer.initialize(symx)
#         # symx=self.symmetrizer.symmetrize_input(symx)
        
#         out_mod=self.net_modulant(symx,z)
#         out_env=self.net_envelope(symx,z)
#         out=torch.exp(out_env) * out_mod
#         # out=self.symmetrizer.antisymmetrize_output(out)
#         return out

# class MLPDecoder(nn.Module):
#     def __init__(self,input_dim,num_layers=2, hidden_dim=3, output_dim=1,pe_type=None, pe_dim=32):
#         super(MLPDecoder,self).__init__()



#     def random_fourier_encoding(self, x):

#         freqs = self.rand_freqs.reshape(1, 1, -1, 3) * (self.D // 2)
#         kx_ky_kz = x[..., None, :] * freqs
#         k = kx_ky_kz.sum(-1)
#         s = torch.sin(k)
#         c = torch.cos(k)
#         x_encoded = torch.cat([s, c], -1)
#         return x_encoded



class Sine(nn.Module):
    def __init__(self, w0=20.0):
        """
        Initialization of sine nonlinearity.

        Parameters
        ----------
        w0: float
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0 * input)

class Homo(nn.Module):
    #def __init__(self,hidden_dim=256, num_layers=2,w0=16.,w=16.,out_dim=1,in_dim=3):
  def __init__(self,in_dim=3,hidden_dim=256,out_dim=2,layers_num=2,activation='SIREN',w0=30.,w=20.,pe=False,num_encoding=6):    
    super(Homo, self).__init__()
    # self.net=SirenNet(dim_in=in_dim,dim_hidden=hidden_dim,dim_out=out_dim,num_layers=layers_num,w0_initial=w0)
    self.in_dim=in_dim
    self.pe=False
    if pe:
      self.pe=True
      self.PE=PositionalEncoding(num_encoding_functions=num_encoding)
      self.in_dim=in_dim * num_encoding *2 

    if activation=='SIREN':
      self.net=SirenNet(dim_in=self.in_dim,dim_hidden=hidden_dim,dim_out=out_dim,num_layers=layers_num,w0_initial=w0,w0=w)
    else:
      self.net=Naive_MLP(in_dim=self.in_dim,hidden_dim=hidden_dim,out_dim=out_dim,hidden_layers=layers_num)
    
    # self.net=Naive_MLP(in_dim=in_dim,hidden_dim=hidden_dim,out_dim=out_dim,hidden_layers=layers_num,activation=activation,w0_initial=w0)
   
  def forward(self,coords):
    if self.pe:
      coords=self.PE(coords)
    out=self.net(coords)
    return out

class Hetero(nn.Module):
    #def __init__(self,hidden_dim=256, num_layers=2,w0=16.,w=16.,out_dim=1,in_dim=3):
  def __init__(self,latent_in_dim=4,latent_out_dim=256,coord_net_in_dim=3,coord_net_out_dim=256,hidden_dim=256,out_dim=2,layers_num=2,w0=30.,w=1.,pe=False,num_encoding=6,imgsize=128):    
    super(Hetero, self).__init__()
    # self.net=SirenNet(dim_in=in_dim,dim_hidden=hidden_dim,dim_out=out_dim,num_layers=layers_num,w0_initial=w0)
    self.pe=False
    if pe:
      self.pe=True
      self.PE=PositionalEncoding(num_encoding_functions=num_encoding)
      self.coords_in_dim=coord_net_in_dim * num_encoding *2 
      self.net=Naive_MLP(in_dim=self.coords_in_dim+latent_in_dim,hidden_dim=hidden_dim,out_dim=out_dim,hidden_layers=layers_num)

    else:
      self.net=SirenNet(dim_in=coord_net_in_dim+latent_in_dim,dim_hidden=hidden_dim,dim_out=out_dim,num_layers=layers_num,w0_initial=w0,w0=w)
      self.wrapper_net=SirenWrapper(self.net,latent_in_dim,192,192)
    # self.net=SirenNet(dim_in=coord_net_out_dim+latent_out_dim,dim_hidden=hidden_dim,dim_out=out_dim,num_layers=layers_num,w0_initial=w0,w0=w)

    # self.coord_layers=SirenNet(dim_in=coord_net_in_dim,dim_hidden=hidden_dim,dim_out=coord_net_out_dim,num_layers=layers_num,w0_initial=w0,w0=w)

    # self.latent_layers=Naive_MLP(in_dim=latent_in_dim,hidden_dim=hidden_dim,out_dim=latent_out_dim,hidden_layers=layers_num,activation='gelu')

    # self.net=Naive_MLP(in_dim=in_dim,hidden_dim=hidden_dim,out_dim=out_dim,hidden_layers=layers_num,activation=activation,w0_initial=w0)
  def forward(self,coords,z):
    if self.pe:
        coords=self.PE(coords)
    #latent_out=self.latent_layers(z)
    #latent_out=self.latent_layers(z).unsqueeze(1).unsqueeze(1)
    #coord_out=self.coord_layers(coords)
    #cat_input=torch.cat( (coords,z.expand([coords.size(0),coords.size(1),coords.size(2),z.size(-1)])) , dim=-1 )
    #out=self.net(cat_input)
    #out=self.wrapper_net(cat_input)
    out=self.wrapper_net(coords,z)
    #out=self.net( torch.cat( (coord_out,latent_out.expand([coord_out.size(0),coord_out.size(1),coord_out.size(2),latent_out.size(-1)])) , dim=-1 ))
    return out

    

class Naive_MLP(nn.Module):
  def __init__(self,in_dim,out_dim,hidden_layers=2,hidden_dim=256,activation='tanh',last_activation='gelu',w0_initial=30.):
    super(Naive_MLP,self).__init__()
    layers=[]
    


    layers.append(nn.Linear(in_dim,hidden_dim))
    activation_layer=name_2_activation(activation)
    if activation=='SIREN':
      layers.append(Sine())
    else:
      layers.append(activation_layer)

    # else:
    #   layers.append(activation)

    for i in range(hidden_layers):
      layers.append(nn.Linear(hidden_dim,hidden_dim))      
      layers.append(activation_layer)      
      # if activation=='SIREN':
      #   layers.append(Sine())
      # else:
      #   layers.append(activation)

    layers.append(nn.Linear(hidden_dim,out_dim))


    # dont use final activation
    if last_activation is not None:
        layers.append(name_2_activation(last_activation)) 
      # if activation=='SIREN':
      #   layers.append(Sine())
      # else:
      #   layers.append(last_activation)


    self.layers=nn.Sequential(*layers)

  def forward(self,x):
    out=self.layers(x)
    return out   





class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=False, log_sampling=True, normalize=False,
                 input_dim=3, gaussian_pe=True, gaussian_variance=0.5):
        """
        Initilization of a positional encoder.

        Parameters
        ----------
        num_encoding_functions: int
        include_input: bool
        log_sampling: bool
        normalize: bool
        input_dim: int
        gaussian_pe: bool
        gaussian_variance: float
        """
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None

        if self.gaussian_pe:
            # this needs to be registered as a parameter so that it is saved in the model state dict
            # and so that it is converted using .cuda(). Doesn't need to be trained though
            self.gaussian_weights = nn.Parameter(gaussian_variance * torch.randn(input_dim*num_encoding_functions, input_dim),
                                                 requires_grad=False)

        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(
                    0.0,
                    self.num_encoding_functions - 1,
                    self.num_encoding_functions)
            else:
                self.frequency_bands = torch.linspace(
                    2.0 ** 0.0,
                    2.0 ** (self.num_encoding_functions - 1),
                    self.num_encoding_functions)

            if normalize:
                self.normalization = torch.tensor(1 / self.frequency_bands)

    def forward(self, tensor) -> torch.Tensor:
        """
        Apply positional encoding to the input.

        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).

        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """

        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:
                    if self.normalization is not None:
                        encoding.append(self.normalization[idx] * func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)


class Build_Encoder(nn.Module):
  def __init__(self,backbone='resnet',basename='resnet18',rotation_dim=6,shift_dim=2,in_channels=1,activation='tanh',last_activation='tanh',image_size=64):
    super(Build_Encoder,self).__init__()
    assert backbone=='resnet' or backbone=='convnext' or backbone=='pdo' or backbone=='mlp' or backbone=='swin' or backbone=='mobile' or backbone=='efficient' or backbone=='maxvit'
    self.flatten=False
    if backbone=='resnet':
        self.feature_extractor=Res2D(basename,in_channels)
        if basename=='resnet18' or basename=='resnet34':
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
          self.feature_extractor=Mobile(basename,in_channels=in_channels)
    elif backbone=='efficient':
        self.middle_dim=1280
        self.feature_extractor=Efficient(basename,in_channels)

    #self.shared_layers=Naive_MLP(self.middle_dim,256,2)



    self.shared_layers=Naive_MLP(self.middle_dim,256,0)

    self.rotation_layers=Naive_MLP(256,rotation_dim,0,activation=activation,last_activation=last_activation)
    if shift_dim!=0:
      self.shift_layers=Naive_MLP(256,shift_dim,0,activation=activation,last_activation=last_activation)
 

    # self.rotation_layers=Naive_MLP(self.middle_dim,rotation_dim,0,activation=activation,last_activation=last_activation)
    # if shift_dim!=0:
    #   self.shift_layers=Naive_MLP(self.middle_dim,shift_dim,0,activation=activation,last_activation=last_activation)

    self.shift_dim=shift_dim
  def forward(self,x):
    if self.flatten :
        x=x.view(-1,self.flatten_size)

    features=self.feature_extractor(x)
    shared_features=self.shared_layers(features)
    # shared_features=self.feature_extractor(x)

    rot=self.rotation_layers(shared_features)    
    if self.shift_dim==0:
      return rot
    else:
      shift=self.shift_layers(shared_features)
      return torch.cat([rot,shift],-1)


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
    base_model=mobilenet.mobilenet_v3_small()
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
    if basename=='resnet18':
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
      base_model=convnext.convnext_tiny()

    #1. replace  input channel
    base_model.features[0][0]=nn.Conv2d(in_channels,96,kernel_size=(4,4),stride=(4,4))
    #2. replace output layer
    base_model.classifier[2]=nn.Identity()

    self.layers=base_model

  def forward(self,x):
    return self.layers(x)

