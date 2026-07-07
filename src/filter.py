import mrcfile
import numpy as np


class filtermethod(object):

  @staticmethod
  def read_mrc(filename):
    mrc=mrcfile.open(filename,permissive=True)
    data=mrc.data
    arr=np.zeros(shape=data.shape,dtype=data.dtype)
    arr[:]=data[:]
    data=0 
    return arr
  @staticmethod
  def add_gaussian(image,ratio=1):
    gaussian=np.array(np.random.normal(np.mean(image),ratio*np.std(image),image.shape),dtype=np.float32)
    return gaussian+image



  
  @staticmethod
  def gen_mask(image_size,radius=12):
    x,y=np.ogrid[0:image_size,0:image_size]
    center_x=(image_size-1)/2
    center_y=(image_size-1)/2
    mask=(x-center_x)**2+(y-center_y)**2<=radius**2
    return np.array(mask,dtype=np.int32)
  
  @staticmethod
  def write_file(imgdata,filename):
  #print(imgdata.shape)
  #print('writing...'+filename) 
    with mrcfile.new(filename,overwrite=True) as mrc:
      mrc.set_data(imgdata)
      mrc.close()

  @staticmethod
  def lowpass(input_image,cutoff):
    image_size=input_image.shape[0]
    ft_radius=image_size/(2*cutoff)
    img_dft=np.fft.fft2(input_image)
    dft_shift=np.fft.fftshift(img_dft)
    dft_mask=filtermethod.gen_mask(image_size)
    dft_filter=dft_mask*dft_shift
    idft_shift=np.fft.ifftshift(dft_filter)
    iimg=np.abs(np.fft.ifft2(idft_shift))
    return np.float32(iimg)


'''
ori=read_mrc('test_image1_rot10.mrc')
ori_min=np.min(ori)
ori_max=np.max(ori)

norm_ori=(ori-ori_min)/(ori_max-ori_min)

filtered_image=lowpass(norm_ori,20)
write_file(filtered_image,'lp20.mrc')
'''
