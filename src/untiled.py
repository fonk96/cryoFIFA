

import mrcfile
import numpy as np
import os
from multiprocessing import Process,Manager,Value,Pool
from glob import glob
import torch
from functools import partial
from torch.utils.data import Dataset
from filter import filtermethod as fm
from PIL import Image
from multiprocessing import Process,Manager,Value,Pool
import lmdb
import sigproc
#import fourier
import ctypes

import sys 

class ParticlesDataset(Dataset):
  def __init__(self,data_tensor,label_tensor):
    self.data_tensor=data_tensor
    self.label_tensor=label_tensor
  def __getitem__(self,index):
    return self.data_tensor[index],self.label_tensor[index]
  def __len__(self):
    return self.data_tensor.size(0)


def image_duplicator(input_images,factor=2,rot_manner=False):
    """
    input_images: [batch,imgsize,imgsize]
    """
    imgsize=input_images.shape[-1]
    input_images=input_images.reshape(-1,imgsize,imgsize)
    assert factor in [2, 4, 8], "Only 2,4,8 symmetric_factor supported"
    if factor == 2:
        if rot_manner:
          copy_img=np.rot90(input_images,2,[-2,-1])
        else:
          copy_img=np.flip(input_images,[-1])
        img_out = np.cat((input_images,copy_img),0)
    elif factor==4 or factor==8 :
        copy_img1=np.rot90(input_images,1,[-2,-1])
        copy_img2=np.rot90(input_images,2,[-2,-1])
        copy_img3=np.rot90(input_images,3,[-2,-1])

        img_out=np.concatenate((input_images,copy_img1,copy_img2,copy_img3),0)
        if factor==8:
            flips=np.flip(img_out,[-1])
            img_out=np.concatenate((img_out,flips),0)
    return img_out


class Symmetrizer():
    def __init__(self):
        """
        Initialization of a Symmetrizer, to enforce symmetry in Fourier space.
        """
        self.half_space_indicator = None
        self.DC_indicator = None

    def initialize(self, coords):
        self.half_space_indicator = which_half_space(coords)
        self.DC_indicator = where_DC(coords)

    def symmetrize_input(self, coords):
        # Place the "negative" coords in the "positive" half space
        coords[self.half_space_indicator] = -coords[self.half_space_indicator]
        return coords

    def antisymmetrize_output(self, output):
        # Flip the imaginary part on the "negative" half space and force DC component to be zero
        batch_sz = output.shape[0]
        N = output.shape[1]
        output = output.reshape(batch_sz, N, -1, 2)
        # output.shape = Batch, N, channels, 2
        channels = output.shape[2]
        half_space = self.half_space_indicator.reshape(batch_sz, N, 1, 1).repeat(1, 1, channels, 2)
        DC = self.DC_indicator.reshape(batch_sz, N, 1, 1).repeat(1, 1, channels, 2)
        output_sym = torch.where(half_space, torch.cat((output[..., 0].unsqueeze(-1),
                                                        -output[..., 1].unsqueeze(-1)), dim=-1), output)
        output_sym_DC = torch.where(DC, torch.cat((output_sym[..., 0].unsqueeze(-1),
                                                torch.zeros_like(output_sym[..., 0].unsqueeze(-1))), dim=-1),
                                 output_sym)
        output_sym_DC = output_sym_DC.reshape(batch_sz, N, -1)
        return output_sym_DC


def which_half_space(coords, eps=1e-6):
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    slab_xyz = (x < -eps)
    slab_yz = torch.logical_and(torch.logical_and(x > -eps, x < eps), y < -eps)
    slab_z = torch.logical_and(torch.logical_and(torch.logical_and(x > -eps, x < eps),
                                                 torch.logical_and(y > -eps, y < eps)), z < -eps)

    return torch.logical_or(slab_xyz, torch.logical_or(slab_yz, slab_z))


def where_DC(coords, eps=1e-6):
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    slab_x = torch.logical_and(x > -eps, x < eps)
    slab_y = torch.logical_and(y > -eps, y < eps)
    slab_z = torch.logical_and(z > -eps, z < eps)

    return torch.logical_and(slab_x, torch.logical_and(slab_y, slab_z))


class Dataloader():
  @staticmethod
  def star_read(file):
    file_p=open(file)
    lines=file_p.read()
    #print(lines)
    file_p=None
    line_sp=lines.split('\n')
    line_sp.remove('')

    return line_sp

  @staticmethod
  def write_file(imgdata,filename):
  #print(imgdata.shape)
  #print('writing...'+filename)
    with mrcfile.new(filename,overwrite=True) as mrc:
      mrc.set_data(imgdata)
      mrc.close()
      
  @staticmethod
  def write_pix_file(imgdata,filename,pix):
    print('writing...'+filename)
    try:
      with mrcfile.open(filename,mode='r+') as mrc:
        mrc.voxel_size=pix
        mrc.set_data(imgdata)
        mrc.close()
    except:
      with mrcfile.new(filename,overwrite=True) as mrc:
        mrc.set_data(imgdata)
        mrc.close()
      with mrcfile.open(filename,mode='r+') as mrc:
        mrc.voxel_size=pix
        #mrc.set_data(imgdata)
        mrc.close()

  @staticmethod
  def cmd_exec(cmd,mode=0):
    if mode==0:
      print(cmd)
      return 0
    elif mode==1:
      res=os.popen(cmd).read()
      return res


  @staticmethod
  def read_mrc(filename,norm=False):
    mrc=mrcfile.open(filename,permissive=True)
    data=mrc.data
    arr=np.zeros(shape=data.shape,dtype=data.dtype)
    arr[:]=data[:]
    data=0
    if norm:
      arr=Dataloader.norm_image(arr)
    return arr



  @staticmethod
  def fast_read_mrc(filename,norm=False):
    return mrcfile.read(filename)





  # @staticmethod
  # def load_and_drop_particles(stack_list,index_dict,line_index_dict,total_len,prefix='',db_path='',dtype=np.float16,norm=False,lowpass=False,lowpassorder=2,lowpasswindow=2):


  #   common_point=0
  #   data_point=0
  #   env=lmdb.open(db_path,map_size=536870912000)

  #   for stack_name in stack_list:
  #     txn=env.begin(write=True)        
  #     ori_data=Dataloader.fast_read_mrc(prefix+stack_name)
  #     if len(ori_data.shape)==2:
  #         #print('only 1 particles in stack '+prefix+stack_name)
  #       # num=line_index_dict[stack_name][0]
  #       # img=np.array(ori_data,dtype=dtype).tobytes()
  #       # txn.put(str(num).encode(),img)    
  #       # txn.commit()
  #       pass    
  #     else:
  #       for i in range(len(index_dict[stack_name])):
  #         img=np.array(ori_data[index_dict[stack_name][i]-1],dtype=dtype).tobytes()
  #         num=line_index_dict[stack_name][i]
  #         txn.put(str(num).encode(),img)

  #     txn.commit()
  #     print("\r", end="", flush=True)
  #     print(str(common_point+1)+'/'+str(len(stack_list))+' stacks written',end="")
  #     common_point+=1         





  @staticmethod
  def load_and_drop_particles_double(stack_list,stack_list_add,index_dict,line_index_dict,total_len,prefix='',db_path='',dtype=np.float16,norm=False,lowpass=False,lowpassorder=2,lowpasswindow=2):


    common_point=0
    data_point=0
    env=lmdb.open(db_path,map_size=536870912000)
    total_stacks=len(stack_list)
    for si in range (total_stacks):
      stack_name=stack_list[si]
      stack_name_add=stack_list_add[si]
      txn=env.begin(write=True)        
      ori_data=Dataloader.fast_read_mrc(prefix+stack_name)
      ori_data_add=Dataloader.fast_read_mrc(prefix+stack_name_add)
      for i in range(len(index_dict[stack_name])):
        img=np.array(ori_data[index_dict[stack_name][i]-1],dtype=dtype) if len(ori_data.shape)==3 else np.array(ori_data,dtype=dtype)
        img_add=np.array(ori_data_add[index_dict[stack_name][i]-1],dtype=dtype) if len(ori_data_add.shape)==3 else np.array(ori_data_add,dtype=dtype)        
        # img=np.array(ori_data[index_dict[stack_name][i]-1],dtype=dtype).reshape(1,ori_data.shape[-1],ori_data.shape[-1]).tobytes() if len(ori_data.shape)==3 else np.array(ori_data,dtype=dtype).reshape(1,ori_data.shape[-1],ori_data.shape[-1]).tobytes()
        # img_add=np.array(ori_data_add[index_dict[stack_name][i]-1],dtype=dtype)reshape(1,ori_data.shape[-1],ori_data.shape[-1]).tobytes() if len(ori_data_add.shape)==3 else np.array(ori_data_add,dtype=dtype).reshape(1,ori_data.shape[-1],ori_data.shape[-1]).tobytes()
        img=img.reshape(1,ori_data.shape[-1],ori_data.shape[-1])
        img_add=img_add.reshape(1,ori_data.shape[-1],ori_data.shape[-1])  
        #print(img.shape,img_add.shape)      
        double_img=np.concatenate([img,img_add]).tobytes()

        num=line_index_dict[stack_name][i]
        txn.put(str(num).encode(),double_img)

      txn.commit()
      print("\r", end="", flush=True)
      print(str(common_point+1)+'/'+str(len(stack_list))+' stacks written',end="")
      common_point+=1         






  @staticmethod
  def load_and_drop_particles(stack_list,index_dict,line_index_dict,total_len,prefix='',db_path='',sym_factor=8,dtype=np.float16):


    common_point=0
    data_point=0
    env=lmdb.open(db_path,map_size=536870912000)

    for stack_name in stack_list:
      txn=env.begin(write=True)        
      ori_data=Dataloader.fast_read_mrc(prefix+stack_name)
      imgsize=ori_data.shape[-1]
      for i in range(len(index_dict[stack_name])):
        img=np.array(ori_data[index_dict[stack_name][i]-1],dtype=dtype) if len(ori_data.shape)==3 else np.array(ori_data,dtype=dtype)

        #img=image_duplicator(img,sym_factor )

        img=img.tobytes()
        num=line_index_dict[stack_name][i]
        txn.put(str(num).encode(),img)

      txn.commit()
      print("\r", end="", flush=True)
      print(str(common_point+1)+'/'+str(len(stack_list))+' stacks written',end="")
      common_point+=1         










        






  @staticmethod
  def load_from_particle_stack(stack_list,index_dict,line_index_dict,total_len,prefix='',norm=False,dtype=np.float32,lowpass=False,lowpassorder=2,lowpasswindow=2):

    test_stack_name=stack_list[0]
    #test_stack=Dataloader.read_mrc(prefix+test_stack_name)
    test_stack=Dataloader.fast_read_mrc(prefix+test_stack_name)
    stack_size=test_stack.shape[-1]

    data=np.zeros([total_len,stack_size,stack_size],dtype=dtype)

    common_point=0
    data_point=0
    for stack_name in stack_list:
        print("\r", end="", flush=True)
        print(str(common_point+1)+'/'+str(len(stack_list))+' stacks loaded',end="")
        #print(stack_name,end="")
        total_data=Dataloader._PoolReader(data,index_dict,line_index_dict,prefix,stack_name,norm,lowpass,lowpasswindow,lowpassorder,dtype)
        common_point+=1



    print('\n')
    return data

  @staticmethod
  def get_column_number(input_str):
    return int(input_str.split('#')[-1])-1


  @staticmethod
  def load_from_particle_dict(particle_dict,class_number,prefix):
    name_list=list(particle_dict.items())
    total_len=len(name_list)
    test_particle=name_list[0]
    test_stack=Dataloader.read_mrc(prefix+test_stack_name)


  @staticmethod
  def get_particle_and_label(f):
    lines=Dataloader.star_read(f)
    start_index=Dataloader.get_start_line(lines)
    particle_dict={}
    for i in range(start_index):
      if lines[i].find('_rlnImageName')!=-1:
        imagename_number=Dataloader.get_column_number(lines[i])
      elif lines[i].find('_rlnClassNumber')!=-1:
        class_number=Dataloader.get_column_number(lines[i])

    print('Reading particles meta...')
    total_len=len(lines)-1
    for i in range(start_index,total_len):
      line_sp=lines[i].split(' ')
      if len(line_sp)>2:
        print("\r", end="", flush=True)
        print(str(i-start_index+1)+'/'+str(total_len-start_index),end="")
        particle_dict[line_sp[imagename_number]]=int(line_sp[class_number])
    print('\n')
    return particle_dict


  @staticmethod
  def get_full_info(f,skip_pose=True,pose_type=2,optics_name='optics',ang_mode=False):

      params_dict={}
      df_index={}
      dfu=[]
      dfv=[]
      dfang=[]

      finding_optics=['_rlnVoltage','_rlnSphericalAberration','_rlnAmplitudeContrast','PixelSize']
      finding_basic=['_rlnDefocusU','_rlnDefocusV','_rlnDefocusAngle']
      basic=['dfu','dfv','dfang']


      if pose_type==2:
        finding_meta=['_rlnDefocusU','_rlnDefocusV','_rlnDefocusAngle','_rlnAnglePsi','_rlnOriginX','_rlnOriginY']
        finding_pose=['_rlnAnglePsi','_rlnOriginX','_rlnOriginY']
        meta=['dfu','dfv','dfang','rot','dx','dy']
        pose=['rot','dx','dy']
        # check this carefully ! Jul 4, 2023
      elif pose_type==3:
        finding_meta=['_rlnDefocusU','_rlnDefocusV','_rlnDefocusAngle','_rlnAngleRot','_rlnAngleTilt','_rlnAnglePsi','_rlnOriginX','_rlnOriginY']
        finding_pose=['_rlnAngleRot','_rlnAngleTilt','_rlnAnglePsi','_rlnOriginX','_rlnOriginY']
        meta=['dfu','dfv','dfang','rot','tilt','psi','dx','dy']
        pose=['rot','tilt','psi','dx','dy']
        print('3d pose type')

      optics_index={'volt':-1,'cs':-1,'ampcont':-1,'apix':-1}
      #optics_dict={'volt':volt,'cs':cs,'ampcont':ampcont,'apix':apix}
      optics=['volt','cs','ampcont','apix']

      imagename_index=0

      pose_index={}
      pose_list=[]

      index_dict={}
      stack_list=[]

      lines=Dataloader.star_read(f)
      # print(lines[:10])
      start_index=Dataloader.get_start_line(lines)
      # print(start_index)
  
      optics_start_line=-1
      optics_end_line=-1
      for i in range(start_index):
        if lines[i].find(optics_name)!=-1:
          optics_start_line=i
          break
  
      if optics_start_line!=-1:
        for i in range (optics_start_line,start_index):
          line_sp=lines[i].split(' ')   
          #print(line_sp)   
          #try:
          #  line_sp[0].index('data_particles')  
          if len(line_sp) >3:
            optics_end_line=i
            break
          #except:
          #  continue

      #print(optics_start_line,optics_end_line)
      if optics_end_line!=-1:
        for i in range(optics_start_line,optics_end_line):
          for j in range(len(finding_optics)):
            if lines[i].find(finding_optics[j])!=-1:
              optics_index[optics[j]]=Dataloader.get_column_number(lines[i])
              # print(lines[i])

      ang_mode=False
      for i in range(start_index):
        if lines[i].find('_rlnImageName')!=-1:
          imagename_index=Dataloader.get_column_number(lines[i])
          #print(imagename_index)
        else:
          for j in range(len(basic)):
            if lines[i].find(finding_basic[j])!=-1:
              df_index[basic[j]]=Dataloader.get_column_number(lines[i])
          if not skip_pose:
            for j in range(len(pose)):
              line_query=lines[i].split(' ')[0]
              if lines[i].find(finding_pose[j])!=-1:
                if not ang_mode and line_query!=finding_pose[j]:
                  print("Using Ang model for "+line_query)
                  ang_mode=True
                pose_index[pose[j]]=Dataloader.get_column_number(lines[i])


      array_index=0
      line_index_dict={}

      #print(optics_index)
      
      if optics_end_line!=-1:
        optic_line=lines[optics_end_line]
        optic_line_split=optic_line.split(' ')
        while 1:
          try:
            optic_line_split.remove('')
          except:
            pass
            break
        #print(optic_line_split)
        for item in optics:
          try:
            params_dict[item]=float(optic_line_split[optics_index[item]])
            print('using '+item+' from meta file: '+str(params_dict[item]))
          except:
            assert False,item+" not found"
            #print('using default '+item+':'+str(params_dict[item]))
            pass
      else:
        for item in optics:
          print('using default '+item+':'+str(params_dict[item]))


      for i in range(start_index,len(lines)):
        line_sp=lines[i].split(' ')
        while 1 :
          try :
            line_sp.remove('')
          except:
            pass
            break
        if len(line_sp)<=1:
            break            
        # print(line_sp)
        # print(line_sp[imagename_index])
        stackname=line_sp[imagename_index].split('@')[-1]

        try:
          if stack_list.index(stackname)!=-1:
            index_dict[stackname].append(int(line_sp[imagename_index].split('@')[0]))
            line_index_dict[stackname].append(array_index)
            array_index+=1

        except:
          stack_list.append(stackname)
          index_dict[stackname]=[]
          line_index_dict[stackname]=[]
          line_index_dict[stackname].append(array_index)
          index_dict[stackname].append(int(line_sp[imagename_index].split('@')[0]))
          array_index+=1

        dfu.append(float(line_sp[df_index['dfu']]))
        dfv.append(float(line_sp[df_index['dfv']]))
        dfang.append(float(line_sp[df_index['dfang']]))

        norm_dx=1
        if ang_mode:
          norm_dx=params_dict['apix']
        # print(norm_dx)
        stackname=line_sp[imagename_index].split('@')[-1]

        if not skip_pose:
          rot=float(line_sp[pose_index['rot']])/180*np.pi
          if pose_type==3:
            tilt=float(line_sp[pose_index['tilt']])/180*np.pi
            psi=float(line_sp[pose_index['psi']])/180*np.pi
          dx=float(line_sp[pose_index['dx']])/norm_dx
          dy=float(line_sp[pose_index['dy']])/norm_dx 
          if pose_type==3:
            pose_list.append([rot,tilt,psi,dx,dy])
          else:
            pose_list.append([rot,dx,dy])

      if skip_pose:
        pose_array=None
      else:
        pose_array=np.array(pose_list)


      params_dict['dfu']=dfu
      params_dict['dfv']=dfv
      params_dict['dfang']=dfang
      # params_dict['volt']=volt
      # params_dict['cs']=cs
      # params_dict['ampcont']=ampcont
      params_dict['bfactor']=0
      # params_dict['apix']=apix


      return stack_list,index_dict,line_index_dict,array_index,params_dict,pose_array

  @staticmethod
  def get_particles_info(stack_list,index_dict,line_index_dict,params_dict,full_pose=None,skip_pose=True):
    full_data_list=[]

    for stack_name in index_dict:
      for stack_index in range(len(index_dict[stack_name])):
        if skip_pose:
          full_data_list.append( [ stack_name,index_dict[stack_name][stack_index], [params_dict ['volt'],params_dict ['cs'],params_dict ['ampcont'],params_dict ['apix'],params_dict['dfu'][line_index_dict[stack_name][stack_index]],params_dict['dfv'][line_index_dict[stack_name][stack_index]],params_dict['dfang'][line_index_dict[stack_name][stack_index]]] ] )
        else:
          full_data_list.append( [ stack_name,index_dict[stack_name][stack_index], [params_dict ['volt'],params_dict ['cs'],params_dict ['ampcont'],params_dict ['apix'],params_dict['dfu'][line_index_dict[stack_name][stack_index]],params_dict['dfv'][line_index_dict[stack_name][stack_index]],params_dict['dfang'][line_index_dict[stack_name][stack_index]]] ,full_pose[[line_index_dict[stack_name][stack_index]]] ] )

    return full_data_list

  @staticmethod
  def load_from_micro_list(mrc_list,lowpass=False,prefix=''):
    data=Manager().list()
    shared_lowpass=Manager().Value('i',1)
    if not lowpass:
      shared_lowpass.value=0
    pool=Pool(processes=16)
    total_len=len(mrc_list)
    common_point=0
    for mrcname in mrc_list:
      mrcname=prefix+mrcname
      #print(mrcname)
      print("\r", end="", flush=True)
      print(str(common_point+1)+'/'+str(total_len-1)+' stacks loaded',end="")
      #print(shared_lowpass.value)
      pool.apply_async(Dataloader._Reader,(mrcname,data,shared_lowpass))
      common_point+=1
    pool.close()
    pool.join()
    data=np.array(data)
    print('\n')
    print('Merging mrcs into one array...')
    print(data.shape)
    return data

    pass


  @staticmethod
  def _PoolReader(data,index_dict,line_index_dict,prefix,stack_name,norm=False,lowpass=False,lowpassorder=2,lowpasswindow=2,dtype=np.float32):


    #particle_stack=Dataloader.read_mrc(prefix+stack_name,norm)
    particle_stack=Dataloader.fast_read_mrc(prefix+stack_name,norm)
    particle_stack=np.array(particle_stack,dtype=dtype)
    #particle_size=particle_stack.shape[-1]
    #share_array=np.frombuffer(shared_array_base,dtype=ctypes.c_double).reshape(total_len,particle_size,particle_size)
    #print(particle_stack)
    #x,y,z=particle_stack.shape
    #new_stack=np.zeros([len(indexs),1,y,z],dtype=particle_stack.dtype)
    indexs=index_dict[stack_name]
    np_index=np.array(indexs,dtype=np.int32)-1

    line_indexs=line_index_dict[stack_name]
    np_line_indexs=np.array(line_indexs,dtype=np.int32)
    stack_len=len(indexs)
    if len(particle_stack.shape)==2:
      center_mrc=particle_stack
    else:
      center_mrc=particle_stack[np_index]
    data[np_line_indexs]=center_mrc
    # for i in range(stack_len):
    #   #This is important for there may be some stacks only contains one particle.
    #   #if particle_stack.shape[0]== particle_stack.shape[-2]:
    #   if len(particle_stack.shape)==2:
    #     center_mrc=particle_stack
    #   else:
    #     center_mrc=particle_stack[indexs[i]-1,:,:]
    #     if lowpass:
    #       radwn=int(center_mrc.shape[-1]/lowpasswindow)
    #       #print(radwn)
    #       center_mrc=Dataloader.lowpass_filter_for_cryoem(center_mrc,radwn,lowpassorder)
    #     #Turn off norm_image
    #     #center_mrc=Dataloader.norm_image(center_mrc)*2-1
    #     #print(np.min(center_mrc),np.max(center_mrc))
    #   #center_mrc=center_mrc*fm.gen_mask(z,radius=64)
    #   #share_data.append(center_mrc)
    #   #print(center_mrc.dtype)
    #   data[line_indexs[i]]=center_mrc
    #   #share_data[line_indexs[i]*particle_size*particle_size:(line_indexs[i]+1)*particle_size*particle_size]=np.array(center_mrc.reshape([1,particle_size*particle_size]))
    #   return stack_len


  @staticmethod
  def get_start_line(input_lines):
    file_i=0
    for i in range(min(100,len(input_lines))):
        if input_lines[i].find('mrc')>0:
          return i
        else:
          continue
      #line_sp=input_lines[i].split(' ')
      #try:
      #  line_sp.remove('')
      #except:
      #  pass
      #try:
        # particle_name=line_sp[0]
        # stackname=particle_name.split('@')[1]
        # file_i=i
        # break
      #except:
      #  continue
    return file_i


  @staticmethod
  def RX(ang):
    return np.asmatrix([1,0,0],[0,np.cos(ang),-np.sin(ang)])


  @staticmethod
  def rot_to_label(input_rot,dim,app=0):
    one_hot_label=np.eye(dim).tolist()
  #!!!
    step=(app+180)/dim
    return one_hot_label[int(input_rot/step)-1]


  @staticmethod
  def long_hot(rot,tilt,dim=6):
    hot=np.eye(dim*dim*2).tolist()
    step=180/dim
    print(' ')
    print(rot,tilt)
    x_ind=int((rot/step)%dim*dim*2  )
    y_ind=int(  (tilt+180) / step  %  (dim*2)  )
    print(x_ind,y_ind)
    return hot[  x_ind +y_ind ]

  @staticmethod
  def main_load(filename=None,avg_file=None,mrcstack=None):
    print('Preparing data...')
    if filename==None:
      return None,None

    lines=Dataloader.star_read(filename)
    stack_list=[]
    all_particles={}

    avg_lines=Dataloader.star_read(avg_file)
    file_i=Dataloader.get_start_line(lines)
    file_j=Dataloader.get_start_line(avg_lines)
    data=[]
    common_point=0
    labels=[]
    ori_mrcs=Dataloader.read_mrc(mrcstack)
    ori_image=[]
    data_size=ori_mrcs[0].shape[0]
    for mrc in ori_mrcs:
      ori_image.append(Image.fromarray(mrc))

    total_len=len(lines)-file_i
    while common_point< total_len:
      line_sp=lines[file_i+common_point].split(' ')
      avg_line_sp=avg_lines[file_j+common_point].split(' ')
      try:
        line_sp.remove('')
        avg_line_sp.remove('')
      except:
        pass
      try:
        particle_name=line_sp[0]
        avg_name=avg_line_sp[0]
        #print(particle_name)
        #print(avg_name)
        stackname=particle_name.split('@')[1]
        print("\r", end="", flush=True)
        print(str(common_point+1)+'/'+str(total_len-1),end="")
      except:
        #print(line_sp)
        break
        #continue
      #if stackname not in stack_list:
      #  stac#9C7135k_list.append(stackname)
      label=[]
      #all_particles[particle_name]=[]
      # shift not in considered.

      '''
      for i in range(2):
        if i==2:
          continue
        label.append(float(line_sp[4+i]))

      '''
      base_bin=3
      #rot=rot_to_label(180+float(line_sp[4]),base_bin*2,app=180)

      # !!!!!! @AUG 9
      tilt=Dataloader.rot_to_label(float(line_sp[5]),base_bin)
      #print(180+float(line_sp[5]),tilt)
      #label=tilt+rot
      #label=rot
      #label=long_hot(float(line_sp[4]),float(line_sp[5]))
      label=tilt
      inplane_rot=float(line_sp[6])

      avg_index=int(avg_line_sp[18])-1
      ori_mrc=ori_image[avg_index].copy()
      rot_mrc=np.array(ori_mrc.rotate(inplane_rot),dtype=np.float32)
      #print(rot_mrc)
      #all_particles[particle_name].append()
      data.append(rot_mrc)
      #write_file(rot_mrc,'test_image'+str(common_point)+'rot'+str(inplane_rot)+'.mrc')
      labels.append(label)
      common_point+=1



    #final_data=np.float32(data).reshape(len(labels),1,data_size,data_size)
    # for spatial-vae
    final_data=np.float32(data).reshape(len(labels),data_size,data_size)

    final_labels=np.array(labels,dtype=np.float32)

    shuffle_index=np.arange(len(labels))
    np.random.shuffle(shuffle_index)

    final_data=final_data[shuffle_index]
    final_labels=final_labels[shuffle_index]
    print('')


    print('data shape:',final_data.shape)
    print('label shape:',final_labels.shape)
    print(np.max(final_data))
    return final_data,final_labels

    #data_tensor=torch.from_numpy(final_data)
    #label_tensor=torch.from_numpy(final_labels)
    #return data_tensor,label_tensor


  @staticmethod
  #@classmethod
  def get_particles_name(filename):
    lines=Dataloader.star_read(filename)
    #print(len(lines))
    start_index=Dataloader.get_start_line(lines)
    #print(start_index)
    imagename_index=0
    index_dict={}
    stack_list=[]
    line_index_dict={}
    indexs=0
    for i in range(start_index):
      if lines[i].find('_rlnImageName')!=-1:
        imagename_index=int(lines[i].split('#')[-1])-1
        break
    for i in range(start_index,len(lines)):
      line_sp=lines[i].split(' ')

      stackname=line_sp[imagename_index].split('@')[-1]
      if len(line_sp)<=1:
        break
      try:
        if stack_list.index(stackname)!=-1:
          index_dict[stackname].append(int(line_sp[imagename_index].split('@')[0]))
          line_index_dict[stackname].append(indexs)
          indexs+=1
          pass
        else:
          print(stack_list.index(stackname))
          print(stackname)
          print('What the fuck else here?')
      except:
         stack_list.append(stackname)
         index_dict[stackname]=[]
         index_dict[stackname].append(int(line_sp[imagename_index].split('@')[0]))
         line_index_dict[stackname]=[]
         line_index_dict[stackname].append(indexs)
         indexs+=1

    return stack_list,index_dict,line_index_dict,indexs

  @staticmethod
  def load_DD_pose(filename):
    lines=Dataloader.star_read(filename)
    pose_index={}
    start_index=Dataloader.get_start_line(lines)
    pose_list=[]

    for i in range(start_index):
      if lines[i].find('_rlnAnglePsi')!=-1:
        pose_index['rot']=int(lines[i].split('#')[-1])-1
      elif lines[i].find('_rlnOriginX')!=-1:
        pose_index['dx']=int(lines[i].split('#')[-1])-1
      elif lines[i].find('_rlnOriginY')!=-1:
        pose_index['dy']=int(lines[i].split('#')[-1])-1

    for i in range(start_index,len(lines)):
      line_sp=lines[i].split(' ')

      if len(line_sp)<=1:
          break
      rot=float(line_sp[pose_index['rot']])/180*np.pi
      dx=float(line_sp[pose_index['dx']])
      dy=float(line_sp[pose_index['dy']])
      pose_list.append([rot,dx,dy])
    pose_array=np.array(pose_list)
    return pose_array


  @staticmethod
  def split_dataset(data,rate=0.8):
    datalen=data.shape[0]
    train_split,test_split=rate,1-rate

    train_split=int(train_split*datalen)
    test_split=int(test_split*datalen)
    data_train=data[:train_split]
    data_test=data[train_split:]

    return data_train,data_test

  @staticmethod
  def render_image(pixel_list):
    #print(len(pixel_list))
    n=m=int(np.sqrt(len(pixel_list)))
  #print(min(pixel_list),max(pixel_list))
    a,b=min(pixel_list),max(pixel_list)
    res=np.zeros([n,m])
    for i in range (n):
        for j in range (m):
            res[i,j]=((pixel_list[i*m+j]-a)/(b-a))*255
    return np.uint8(res)


  def render_image_new(pixel_list):

    n=m=int(np.sqrt(len(pixel_list)))
    res=np.array(pixel_list).reshape(n,m)
    
    a,b=min(pixel_list),max(pixel_list)
    #res=np.zeros([n,m])

    res=((res-a)/(b-a))*255
    return np.uint8(res)


  @staticmethod
  def render_mrc(pixel_array):
    #print(len(pixel_list))
    imgsize=int(np.cbrt(pixel_array.shape[0]))
    #print('\nconverting pixel_list into mrc array')
    array_mrc=pixel_array.reshape(imgsize,imgsize,imgsize)

    a,b=min(pixel_array),max(pixel_array)
    array_mrc=((array_mrc-a)/(b-a))*255
    return np.uint8(array_mrc)


  # @staticmethod
  # def render_stack_mrc(pixel_array):
  #   #print(len(pixel_list))
  #   stack_size=int(np.cbrt(pixel_array.shape[0]))
  #   imgsize=int(np.cbrt(pixel_array.shape[1]))
  #   #print('\nconverting pixel_list into mrc array')
  #   array_mrc=pixel_array.reshape(stack_size,imgsize,imgsize)

  #   a,b=min(pixel_array),max(pixel_array)
  #   array_mrc=((array_mrc-a)/(b-a))*255
  #   return np.uint8(array_mrc)

  @staticmethod
  def _Reader(mrcname,shared_data,lowpass=6):
    #print(mrcname)
    #print(lowpass.value)
    #if not lowpass.value:
    #  print('lowpass')
    stack=Dataloader.read_mrc(mrcname)
    if stack.shape[0]==stack.shape[-2]:
      if not lowpass.value:
        shared_data.append(stack)
      else:
         # invert better ?
         # 255* better ?
         # lowpass tighter ?
        shared_data.append(Dataloader.norm_image(Dataloader.lowpass_filter_for_cryoem(stack)))
        #shared_data.append(1-Dataloader.norm_image(Dataloader.lowpass_filter_for_cryoem(stack)))
    else:
      for slice in stack:
        if not lowpass.value:
          shared_data.append(slice)
        else:
          shared_data.append(1-Dataloader.norm_image(Dataloader.lowpass_filter_for_cryoem(slice)))
          shared_data.append(Dataloader.norm_image(Dataloader.lowpass_filter_for_cryoem(slice)))
    pass
    #print('norm exit.')

  @staticmethod
  def read_from_path(filepath,lowpass=False):

    mrc_list=glob(filepath+'*.mrc')
    data=Dataloader.load_from_micro_list(mrc_list)
    return data


  @staticmethod
  def norm_image(image):
    mi,ma=np.min(image),np.max(image)
    norm_image=(image-mi)/(ma-mi)
    return norm_image

  @staticmethod
  def norm_image_zscore(image,mask=None):
    '''Zero-mean / unit-std normalization computed *inside* `mask` (numpy bool/float,
    same shape as image). Unlike global min-max, one hot voxel no longer sets the scale,
    so solvent sits at ~0 and contrast reflects the bulk of the density. mask=None -> whole box.'''
    if mask is not None:
      sel = np.asarray(mask) > 0.5
      if sel.sum() > 0:
        mu = image[sel].mean(); sd = image[sel].std()
      else:
        mu = image.mean(); sd = image.std()
    else:
      mu = image.mean(); sd = image.std()
    return (image-mu)/(sd+1e-8)

  @staticmethod
  def write_file_apix(imgdata,filename,apix):
    '''Like write_file but stamps the voxel size into the MRC header (the plain
    write_file leaves apix=0, which makes ChimeraX/RELION mis-scale the map).'''
    with mrcfile.new(filename,overwrite=True) as mrc:
      mrc.set_data(imgdata)
      try:
        mrc.voxel_size = float(apix)
      except Exception:
        pass
      mrc.close()

  @staticmethod
  def norm_unsigned(image):
    mi,ma=np.min(image),np.max(image)
    norm_image=(image-mi)/(ma-mi)
    return np.uint8(norm_image*255)

  @staticmethod
  def lp_mask(N=64,radwn=6):
    return  sigproc.lowpass_mask(N, 2, radwn)

  @staticmethod
  def lowpass_filter_for_cryoem(img_gray,radwn=6,lowpassorder=1):


    rD = img_gray
    N = rD.shape[0]
    # 计算低通滤波的mask
    lowpass_mask = sigproc.lowpass_mask(N, 2, radwn, lowpassorder=lowpassorder)
    D = fourier.fft(rD)  # 傅里叶变换
    Cval = 1.0
    dispim = fourier.ifft(lowpass_mask * D * Cval)  # 傅里叶变换

    return dispim

