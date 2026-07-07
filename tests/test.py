
import numpy as np


from dataio import *

from untiled import Dataloader as DL

from utils import *
import sys
import json
from fourier_utils import *

from model_utils import *

from torchvision.transforms.functional import rotate
import time
from new_utils import *

import argparse

parser=argparse.ArgumentParser('')
parser.add_argument('--ctf',action='store_true',default=False)
parser.add_argument('--filters',action='store_true',default=False)
parser.add_argument('--input_volume',type=str,default='')
parser.add_argument('--angpix',type=float,default=2.165)
parser.add_argument('--imgsize',type=int,default=200)
parser.add_argument('--b',type=int,default=20000)
parser.add_argument('--cuda',action='store_true',default=False)
parser.add_argument('--cuda_block',type=int,default=4096)
parser.add_argument('--mask',action='store_true',default=False)
parser.add_argument('--ft',action='store_true',default=False)
parser.add_argument('--shift',action='store_true',default=False)
parser.add_argument('--coords_shift',action='store_true',default=False)
parser.add_argument('--simulate_data',action='store_true',default=False)
parser.add_argument('--input_angle',type=str,default='')
parser.add_argument('--mode',type=str,default='fourier')
parser.add_argument('--input_star',type=str,default='')
parser.add_argument('--downasmple',action='store_true',default=False)
parser.add_argument('--output_prefix',type=str,default='')
args = parser.parse_args()


input_volume=torch_load(args.input_volume,np.float32)
device=torch.device('cuda')
input_volume=input_volume.to(device)
ft_slices=dt(ft2(input_volume).real)


DL.write_file(ft_slices,'fourier_slice.mrc')
sys.exit(1)



ft_volume=ft3(input_volume)
imgsize=input_volume.size(-1)




# helper_x,helper_y=torch.meshgrid(torch.arange(-imgsize/2,imgsize/2),torch.arange(-imgsize/2,imgsize/2))
# helper_x=helper_x.to(device)
# helper_y=helper_y.to(device)   

lowpass_mask_3d=torch.from_numpy(gen_lowpassmask_3d(imgsize,16)).float().to(device)

soft_mask=gen_volume_mask(ft_volume.real,lowpass_mask_3d,mask_threshold=0.75).to(device) 
#gen_mask_projection(soft_mask,rotated_plane_grid,translation_3d,helper_x,helper_y)

# imgsize=32
# device=torch.device('cuda')
# shifts=torch.randint(32,(16,2),device=device)-16
# ori_masks=gen_mask(32,0.25).to(device).expand(16,32,32)

# masks=gen_shiftmasks_torch(64,0.25,shift=shifts,device=device)
# helper_x,helper_y=torch.meshgrid(torch.arange(-imgsize/2,imgsize/2),torch.arange(-imgsize/2,imgsize/2))
# helper_x=helper_x.to(device)
# helper_y=helper_y.to(device)  


# shift_mask=ift2(phase_shift_ft(ft2(ori_masks),shifts,helper_x,helper_y))

# np_masks=shift_masks(64,0.25,shifts)
# DL.write_file(dt(masks),'test_torchshift.mrc')
# DL.write_file(dt(np_masks),'test_npshift.mrc')

# DL.write_file(dt(shift_mask),'shift_mask.mrc')

DL.write_file(dt(soft_mask),'soft_mask.mrc')
sys.exit(1)
# b=args.b
imgsize=args.imgsize
cuda_block=args.cuda_block
input_volume=None
if args.input_volume!='':
  input_volume=torch_load(args.input_volume,np.float32)
  
  imgsize=input_volume.shape[-1]

if args.cuda:
  device=torch.device('cuda')
else:
  device=torch.device('cpu')


if args.simulate_data:
  if args.input_star!='':
    stack_list,index_dict,line_index_dict,total_len,params_dict,full_pose=DL.get_full_info(args.input_star,skip_pose=False,pose_type=3)
    dfu=list2tensor(params_dict['dfu'])
    dfv=list2tensor(params_dict['dfv'])
    dfang=list2tensor(params_dict['dfang'])

    batchs=int(total_len/1000)+1
    for i in range(batchs):
      output_name=args.output_prefix+'newbatch_'+num2str(i,6)+'_downsample.mrc'
      input_pose=full_pose[(i*1000):((i+1)*1000)]
      input_dfu=dfu[(i*1000):((i+1)*1000)]
      input_dfv=dfv[(i*1000):((i+1)*1000)]
      input_dfang=dfang[(i*1000):((i+1)*1000)]
      simulate_data(input_volume=input_volume,input_pose=input_pose,output_name=output_name,save=True,b=len(input_pose),device=device,input_dfu=input_dfu,input_dfv=input_dfv,input_dfang=input_dfang)
    #print(full_pose)
    # for stack in params_dict:
    #   print(params_dict[stack])
    #simulate_data(input_volume=input_volume,save=True,input_theta=)

  sys.exit(1)
if args.mask:
  #full_mask=np.array(gen_mask(imgsize,0.5))
  #full_mask=dt(gen_mask(imgsize,0.5))
  #print(full_mask)
  full_mask=DL.lp_mask(imgsize,imgsize)
  DL.write_file(DL.norm_unsigned(full_mask),'oldlp_mask.mrc')
  sys.exit(1)





if args.ft:

  input_volume=input_volume.view(-1,imgsize,imgsize)
  #stack_len=input_volume.shape[0]
  print(input_volume.shape)
  # real_mask=gen_mask(imgsize,0.5,'2d',True)
  # # print(input_volume.shape,real_mask.shape)
  # # input_volume=torch.mul(input_volume,real_mask)
  # ft_mask=gen_mask(imgsize,0.5,'2d')
  #out_image=torch.mul(ft2(input_volume),ft_mask)
  # rot_angles=open('/mgh2/kongfang/test/ang').read().split('\n')
  # try:
  #   rot_angles.remove('')
  # except:
  #   pass
  # angles=np.zeros(stack_len)
  # for i in range(len(rot_angles)):
  #   #print (input_volume[i].shape)
  #   temp=rotate(input_volume[i].view(1,imgsize,imgsize),-1*float(rot_angles[i]))
  #   input_volume[i]=temp[0]
    #print(temp.shape)
  #   angles[i]=float(rot_angles[i])
  # torch_angles=torch.from_numpy(angles)
  # rot_image=rotate(input_volume,torch_angles)
  # DL.write_file(DL.norm_unsigned(dt(input_volume)),'rot_ori.mrc')  
  #out_image=torch.fft.fftshift(torch.fft.fftn(input_volume,dim=(-2,-1)),dim=(-2,-1))

  out_image=ft2(input_volume)

  DL.write_pix_file(DL.norm_unsigned(dt(out_image)),'ft2.mrc',args.angpix)
  #out_image=torch.mean(out_image,dim=0)
  re_image=ift2(out_image)
  #re_image=torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(out_image,dim=(-2,-1)),dim=(-2,-1)),dim=(-2,-1)).real
  #out_image=out_image.view(-1,imgsize,imgsize)
  #re_image=re_image.view(-1,imgsize,imgsize)

  # out_image=torch.mean(out_image,dim=0)

  # std_profile=compute_rotational_std(out_image)
  # mean_profile=compute_rotational_average(out_image)

  #averaged_image=torch.mul(create_image_from_radial_profile(mean_profile),ft_mask)
  # std_image=torch.mul(create_image_from_radial_profile(std_profile),ft_mask)
  DL.write_pix_file(DL.norm_unsigned(dt(out_image)),'mean_ft2.mrc',args.angpix)

  DL.write_pix_file(DL.norm_unsigned(dt(re_image)),'re.mrc',args.angpix)  
  # DL.write_file(DL.norm_unsigned(dt(averaged_image)),'avg_noiseft.mrc')  
  # DL.write_file(DL.norm_unsigned(dt(std_image)),'std_noiseft.mrc')  
  sys.exit(1)


  # stack_len=input_volume.size(0)
  # for i in range(stack_len):
  #   out_image=ft2(input_volume[i])
  #   #filtered_images=gaussian_filters(input_volume[i])
  #   DL.write_file(DL.norm_unsigned(dt(out_image)),'slice_'+str(i)+'_filtered_mrc')
  #   print_refresh(str(i+1)+'/'+str(stack_len))  

if args.ctf:
  t1=time.time()
  
  dfs=torch.randint(5000,20000,(b,2))
  dfu=dfs[...,0]
  dfv=dfs[...,1]     
  dfang=torch.rand(b)


  ctfs=cuda_create_ctf(imgsize,imgsize,dfu,dfv,dfang)

  t=time.time()-t1

  print(t)
  sys.exit(1)

  ctfs=torch.zeros(b,imgsize,imgsize)

  batch=max(1,int(b/cuda_block))
  if batch*cuda_block<b:
    batch+=1
  finished=0
  for i in range(batch):

    batch_dfs=dfs[i*cuda_block:(i+1)*cuda_block]
    dfu=batch_dfs[...,0]
    dfv=batch_dfs[...,1]   
    batch_dfang=dfang[i*cuda_block:(i+1)*cuda_block].detach().to(device)
    ctfs[i*cuda_block:(i+1)*cuda_block]=create_ctf(imgsize,imgsize,dfu,dfv,batch_dfang).view(batch_dfs.size(0),imgsize,imgsize).detach()
    torch.cuda.empty_cache()
    finished+=batch_dfs.size(0)
    print(str(finished)+'/'+str(b),"\r",end="",flush=True)

  t=time.time()-t1

  print(t)
  sys.exit(1)

if args.filters:
  # imgsize=args.imgsize
  
  input_volume=input_volume.view(-1,1,imgsize,imgsize)
  window_mask=gen_mask(imgsize,0.45)
  #input_volume=torch.mul(input_volume,window_mask)
  gaussian_filters = GaussianPyramid(
                kernel_size=11,
                kernel_variance=0.1,
                num_octaves=3,
                octave_scaling=10
            )
  # lowpassed_volume=torch_lowpass(input_volume)
  # DL.write_file(DL.norm_unsigned(dt(lowpassed_volume)),'lowpass_filtered_mrc')
  # stack_len=input_volume.size(0)

  # image_slice=input_volume[-1].view(-1,1,imgsize,imgsize)

  # lowpassed=apply_lowpass(image_slice,8).view(-1,imgsize,imgsize)
  # lowpassed=torch.mul(window_mask,lowpassed)
  #print(lowpassed.size())
  #lowpassed=torch_lowpass(input_volume)
  # DL.write_file(DL.norm_unsigned(dt(lowpassed)),'reallowpass.mrc')

  gaussian=gaussian_filters(input_volume)[:,-1,...]
  gaussian=torch.mul(window_mask,gaussian)

  DL.write_file(DL.norm_unsigned(dt(gaussian)),'gaussian.mrc')
  # for i in range(stack_len):
    
  #   filtered_images=gaussian_filters(input_volume[i])
  #   DL.write_file(DL.norm_unsigned(dt(filtered_images)),'slice_'+str(i)+'_filtered_mrc')
  #   print_refresh(str(i+1)+'/'+str(stack_len))
  sys.exit(1)





# masks=[]
# new_masks=[]
# for i in range(4):
#   radius=(i+1)*8./64.
#   masks.append(gen_mask(64,radius))
#   new_masks.append(gen_mask(64,radius))
#   #DL.write_file(dt(masks[i]),'orimask_'+str(i+1)+'.mrc')
# for i in range(3):
#   new_masks[i+1]=new_masks[i+1]-masks[i]
#   DL.write_file(dt(new_masks[i+1]),'orimask_'+str(i+2)+'.mrc')
# sys.exit(1)

# slices=ft2(ori_volume)

# # print(slices.shape)
# imgsize=ori_volume.shape[-1]
# b=ori_volume.shape[0]
# shift_x=torch.ones((b,1))*imgsize/4
# # shift_y=torch.zeros_like(shift_x)
# #shift_y=torch.ones((b,1))*imgsize/4
# shift_y=torch.ones((b,1))*0
# shift=torch.cat((shift_y,shift_x),dim=-1)
# # print(projections.size())
# # print(shift.size())

# # DL.write_file(dt(slices.real),'ft_shifts.mrc')


# slices=phase_shift_ft(slices,shift)

# DL.write_file(dt(ift2(slices).real),'test_shift_masks.mrc')
# sys.exit(1)

# b=slices.size(0)
# imgsize=slices.size(-1)
# imgsize=128
# b=6
#
# b=1
# imgsize=64

# dfs=torch.randint(5000,20000,(b,2))
# dfu=dfs[...,0]
# dfv=dfs[...,1]
# dfang=torch.rand(b)
# CTF=create_ctf(imgsize,imgsize,dfu,dfv,dfang).view(b,imgsize,imgsize)
#CTF=compute_ctf(imgsize,imgsize,dfu,dfv,dfang).view(b,imgsize,imgsize)

# DL.write_file(DL.norm_unsigned(dt(CTF)),'new_ctf.mrc')



#test_volume=torch_load('/mgh/ycy1/Zhuangqi/P28/J264/downsample/batch_000000_downsample.mrc')[0].float()
test_volume=torch_load('restore.mrc',np.float32)
imgsize=test_volume.size(-1)
#b=test_volume.size(0)


rot_image=gaussian_filters(test_volume)
DL.write_file(DL.norm_unsigned(dt(rot_image)),'gaussian_filtered.mrc')
sys.exit(1)
for i in range(3):
  rot_image=torch.rot90(test_volume,i+1,[-2,-1])
  rot_image=gaussian_filters(rot_image)
  #rot_image=image_duplicator(rot_image,2).view(2,imgsize,imgsize)


  DL.write_file(DL.norm_unsigned(dt(rot_image)),'rot'+str(i+1)+'.mrc')

sys.exit(1)
#test_volume=torch_load('cryosparc_P73_J39_map.mrc',np.float32)
# imgsize=test_volume.size(-1)
# test_volume=test_volume.view(1,imgsize,imgsize)
#mask=gen_mask(imgsize)
# b=6


# slices=ft2(test_volume)
dfs=torch.randint(5000,20000,(b,2))
dfu=dfs[...,0]
dfv=dfs[...,1]
dfang=torch.rand(b)
CTF=create_ctf(imgsize,imgsize,dfu,dfv,dfang).view(b,imgsize,imgsize)
#CTF=compute_ctf(imgsize,imgsize,dfu,dfv,dfang).view(b,imgsize,imgsize)

DL.write_file(DL.norm_unsigned(dt(CTF)),'new_ctf.mrc')

# sys.exit(1)
# print(slices.size())
#sys.exit(1)
slices=slices*CTF

DL.write_file(DL.norm_unsigned(dt(ift2(slices))),'Cr.mrc')

sys.exit(1)


shift=torch.randint(int(-imgsize/10),int(imgsize/10),(b,2))


#SNR=0.5


slices=phase_shift_ft(slices,shift)

slices=add_gaussian_noise_fourier_space(slices,0.5)

slices=ift2(slices)

#DL.write_file(DL.norm_unsigned(dt(slices)),'noisy_shift_proj.mrc')
#DL.write_file(DL.norm_unsigned(dt(slices)),'noisy_proj.mrc')

#DL.write_file(DL.norm_unsigned(dt(slices)),'simu_proj.mrc')
# DL.write_file(DL.norm_unsigned(dt(slices)),'ctf_proj.mrc')
sys.exit(1)
#ctf_file=open('simu_ctf','w')
dfu=dt(dfu.view(b))
dfv=dt(dfv.view(b))
dfang=dt(dfang)

ctf_dict={}
ctf_dict['dfu']=dfu.tolist()
ctf_dict['dfv']=dfv.tolist()
ctf_dict['dfang']=dfang.tolist()

# simu_ctf

with open('ctf_only','w') as f:
	json.dump(ctf_dict,f)

sys.exit(1)


#test_volume=torch_load('cryosparc_P73_J22_class_00_final_volume.mrc',np.float32)
test_volume=torch_load('cryosparc_P73_J39_map.mrc',np.float32)
imgsize=test_volume.size(-1)
mask=gen_mask(test_volume.size(-1),0.5,'3d')



test_volume=test_volume*mask



simulate_data(test_volume,3000,True,'simu64_rot_shift.mrc',False,True,'fourier',mask)
sys.exit(1)



device=init_device(0)
b=1
coords=gen_coords_grid(imgsize,'2d',False).to(device)
coords=coords.expand(b,coords.size(0),coords.size(1))


#=================fourier slice test=========

#volume=ft3(test_volume).to(device).expand(b,imgsize,imgsize,imgsize)
volume=ft3(test_volume).to(device).view(b,imgsize,imgsize,imgsize)
slices=slicer(volume,coords,'fourier').view(b,imgsize,imgsize)
projections=ift2(slices)
DL.write_file(DL.norm_unsigned(dt(projections)),'projections.mrc')


#================phase shift test============

shift=torch.randint(int(-imgsize/8),int(imgsize/8),(b,2))
# print(projections.size())
# print(shift.size())
slices=phase_shift_ft(slices,shift)


#shifted_image=ift2(slices).view(imgsize,imgsize)
#DL.write_file(DL.norm_unsigned(dt(slices)),'single_shifted.mrc')


#=================CTF modulation test=========


dfu=dfv=torch.tensor([10000]).view(b,1)
dfang=torch.zeros((b,1))
CTF=create_ctf(imgsize,imgsize,dfu,dfv,dfang).view(b,imgsize,imgsize).to(device)

#DL.write_file(DL.norm_unsigned(dt(CTF)),'Pure_CTF.mrc')

slices=CTF*slices

slices=ift2(slices)

DL.write_file(DL.norm_unsigned(dt(slices)),'proj_ctf.mrc')




