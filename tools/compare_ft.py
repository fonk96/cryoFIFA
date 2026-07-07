import numpy as np
from scipy.ndimage import distance_transform_edt

from dataio import *


def radial_variance(I, center=None):
    # 计算图像中心，默认为图像中心
    if center is None:
        center = (I.shape[0]//2, I.shape[1]//2)

    # 计算每个像素到图像中心的距离
    dist = distance_transform_edt(np.logical_not(I), return_distances=True, origin=center)

    # 将距离矩阵转换为同心圆标签
    r = np.round(dist).astype(int)

    # 初始化输出矩阵S
    S = np.zeros_like(I)

    # 对于每一个可能的半径值（从0到N）
    for i in range(I.shape[0]//2 + 1):
        # 获取该半径下的所有像素点
        mask = r == i
        # 计算这些像素点的方差，并赋值给输出矩阵S对应的位置
        S[mask] = np.var(I[mask])

    # 由于我们只计算了到中心的一半距离，所以对称复制另一半
    S = np.flipud(np.fliplr(S))

    return S
volume_a=torch_load('/data2/kongfang/CS-installation-test/J16/J16_002_volume_map.mrc',np.float32)

# volume_b=torch_load('rec_200.mrc',np.float32)

#fa=ft3(volume_a)

slices,theta=simulate_data(volume_a,100,True)
#print(slices)

# volume_slice=fa[160,160,160]
# rsd=radial_variance(volume_slice)



# print(fa.real,fa.imag)
# fb=ft3(volume_b)


# DL.write_file(dt(fa),'r100ft.mrc')
# DL.write_file(dt(fb),'r200ft.mrc')

# fb[-60:60,-60:60,-60:60]=fa[-60:60,-60:60,-60:60]


# fc=ift3(fb)

# DL.write_file(dt(fb),'r200ft_new.mrc')

# DL.write_file(dt(fc),'r200_new.mrc')
