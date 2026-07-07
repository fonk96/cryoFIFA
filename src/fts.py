
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def rotate_3d(x,y,z):
  rot_x=np.zeros([3,3])
  rot_y=np.zeros([3,3])
  rot_z=np.zeros([3,3])

  rot_x[2,2] = 1
  rot_x[1,1]=np.cos(x)
  rot_x[1,0]=-np.sin(x)
  rot_x[0,1]=np.sin(x)
  rot_x[0,0]=np.cos(x)

  rot_y[1,1] = 1
  rot_y[0,0]=np.cos(y)
  rot_y[2,0]=-np.sin(y)
  rot_y[0,2]=np.sin(y)
  rot_y[2,2]=np.cos(y)


  rot_z[2,2] = 1
  rot_z[1,1]=np.cos(z)
  rot_z[1,0]=-np.sin(z)
  rot_z[0,1]=np.sin(z)
  rot_z[0,0]=np.cos(z)

  return rot_x,rot_y,rot_z
def syn(arr):
  imgsize=arr.shape[-2]
  print("Syn data...")
  full_data=np.zeros([1000,imgsize,imgsize],dtype=arr.dtype)
  full_pose=2*np.pi*(np.random.random([1000,3])-0.5)
  pass

  
def rotate__patch(x,y,z):
  rot_x=np.zeros([3,3])
  rot_y=np.zeros([3,3])
  rot_z=np.zeros([3,3])

  rot_x[2,2] = 1
  rot_x[1,1]=np.cos(x)
  rot_x[1,0]=-np.sin(x)
  rot_x[0,1]=np.sin(x)
  rot_x[0,0]=np.cos(x)

  rot_y[1,1] = 1
  rot_y[0,0]=np.cos(y)
  rot_y[2,0]=-np.sin(y)
  rot_y[0,2]=np.sin(y)
  rot_y[2,2]=np.cos(y)


  rot_z[2,2] = 1
  rot_z[1,1]=np.cos(z)
  rot_z[1,0]=-np.sin(z)
  rot_z[0,1]=np.sin(z)
  rot_z[0,0]=np.cos(z)

  return rot_x,rot_y,rot_z

def torch_ft(image):
  # return torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(image))))
  return torch.fft.fftshift(torch.fft.fft2(image))
  # return torch.fft.fft2(image)

def torch_ft_3d(image):
  # return torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(image))))
  return torch.fft.fftshift(torch.fft.fftn(image))
  # return torch.fft.fft2(image)
  pass



def rotate(self, img, theta):
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    rotT = torch.stack([cos, sin, -sin, cos], 1).view(-1, 2, 2)
    grid = self.model.lattice.coords[:,0:2] @ rotT
    grid = grid.view(-1, self.D, self.D, 2)
    return F.grid_sample(img, grid)



