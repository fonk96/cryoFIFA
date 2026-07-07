import torch
from torch.utils.data import Dataset
from untiled import Dataloader as DL
import numpy as np
from utils import *
import time
import lmdb
class StarDataSet(Dataset):
  def __init__(self,full_data_list,data_path,imgsize=128,skip_pose=True,pose_type=3):
    self.full_data_list=full_data_list
    self.skip_pose=skip_pose
    self.data_path=data_path
    self.imgsize=imgsize
  def __len__(self):
    return len(self.full_data_list)
  
  def __getitem__(self,idx):
    t1=time.time()
    imgdata=np.array(DL.fast_read_mrc(self.data_path+self.full_data_list[idx][0]))
    particle_index=self.full_data_list[idx][1]
    if len(imgdata.shape)==2:
      mrcdata=imgdata
    else:
      mrcdata=imgdata[particle_index-1,:,:]

    mrcdata=torch.tensor(mrcdata)
    #print('get_mrc_cost:',time.time()-t1)
    t1=time.time()
    ctf_params={'volt':0,'cs':0,'ampcont':0,'apix':0,'dfu':0,'dfv':0,'dfang':0}
    ctf_index=0
    for ctf_item in ctf_params:
      ctf_params[ctf_item]=self.full_data_list[idx][2][ctf_index]
      ctf_index+=1
    ctf=get_ctf(ctf_params,self.imgsize,self.imgsize)
    #print('get_ctf_cost:',time.time()-t1)
    t1=time.time()   
    pose = None if self.skip_pose else torch.tensor(self.full_data_list[idx][3])
    #print('get_pose_cost:',time.time()-t1)    
    if self.skip_pose:
      return mrcdata,ctf
    else:
      return mrcdata,ctf,pose





def crop_input(input_tensor,output_radius):
  '''
    Input size should be [ b ,1 ,imgsize ,imgsize ]
    Output size will be [ b, 1 ,croped_size , croped_size ]
  '''
  #b=input_tensor.size(0)
  #channel=input_tensor.size(1)
  ori_size=input_tensor.size(-1)
  imgcenter=ori_size//2

  #new_tensor=torch.zeros(b,channel,2*output_radius,2*output_radius,device=input_tensor.device)
  #new_tensor=torch.zeros(2*output_radius,2*output_radius)

  input_ft=ft2(input_tensor)

  ft_slice=input_ft[imgcenter-output_radius:imgcenter+output_radius,imgcenter-output_radius:imgcenter+output_radius]
  return ift2(ft_slice).real
  # try:
  #   new_tensor[...]=ift2(ft_slice).real
  # except:
  #   print(ft_slice.size())
  #   print(input_ft.size())
  # return new_tensor



class LmdbDataSet(Dataset):
  def __init__(self,full_data_list,data_path,db,imgsize=128,dtype=np.float16,skip_pose=True,pose_type=3,norm=False,double=False,sym_factor=8,downsample=False,downsample_size=128,with_name=False):
    self.full_data_list=full_data_list
    self.skip_pose=skip_pose
    self.data_path=data_path
    self.imgsize=imgsize
    self.db=lmdb.open(db,lock=False,map_size=53687091200)
    self.norm=norm
    self.dtype=dtype
    self.double=double
    self.sym_factor=sym_factor
    self.downsample=downsample
    self.downsample_size=downsample_size
    self.with_name=with_name
  def __len__(self):
    return len(self.full_data_list)
  
  def __getitem__(self,idx):
    t1=time.time()

    tnx=self.db.begin(write=False)
    mrcdata=torch.from_numpy( np.frombuffer(tnx.get(str(idx).encode()),dtype=self.dtype).reshape(self.imgsize,self.imgsize) )
    # mrcdata=torch.from_numpy( np.frombuffer(tnx.get(str(idx).encode()),dtype=self.dtype).reshape(1,self.imgsize,self.imgsize) )    if not self.double else torch.from_numpy( np.frombuffer(tnx.get(str(idx).encode()),dtype=self.dtype).reshape(2,self.imgsize,self.imgsize) )  

    if self.downsample:
      mrcdata=crop_input(mrcdata)
    #mrcdata=image_duplicator(mrcdata,self.sym_factor)

    # img duplicator here. 

    ctf_params={'volt':0,'cs':0,'ampcont':0,'apix':0,'dfu':0,'dfv':0,'dfang':0}
    ctf_index=0
    for ctf_item in ctf_params:
      ctf_params[ctf_item]=self.full_data_list[idx][2][ctf_index]
      ctf_index+=1

    if self.downsample:
      ctf_params['apix']=ctf_params['apix']*self.imgsize/self.downsample_size


    ctf= get_ctf(ctf_params,self.downsample_imgsize,self.downsample_imgsize) if self.downsample else get_ctf(ctf_params,self.imgsize,self.imgsize)

    #print('get_ctf_cost:',time.time()-t1)
    #t1=time.time()   
    pose = None if self.skip_pose else torch.tensor( np.array(self.full_data_list[idx][3], dtype=self.dtype ) ).reshape(-1)
    #print('get_pose_cost:',time.time()-t1)    
    particles_name=str(self.full_data_list[idx][1])+'@'+str(self.full_data_list[idx][0])


    if self.with_name:
      if self.skip_pose:
        return mrcdata,ctf,ctf_params,particles_name
      else:
        return mrcdata,ctf,pose,ctf_params,particles_name
    else:
      if self.skip_pose:
        return mrcdata,ctf
      else:
        return mrcdata,ctf,pose
