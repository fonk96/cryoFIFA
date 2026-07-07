from utils import *
from untiled import Dataloader as DL
from new_model_utils import GaussianPyramid
gaussian_filters = GaussianPyramid(
                kernel_size=7,
                kernel_variance=0.1,
                num_octaves=5,
                octave_scaling=10.,
                include_input=False)
# input_mrcs=torch.from_numpy(DL.fast_read_mrc('/data2/kongfang/CS-installation-test/J115/downsample/batch_000000_downsample.mrc'))

# input_mrcs=torch.from_numpy(DL.fast_read_mrc('/data2/kongfang/data/Klat/J68/downsample/batch_000000_downsample.mrc'))



# input_mrcs=torch.from_numpy(DL.fast_read_mrc('/data2/kongfang/data/Wanruixue_Bact/Extract/20171224_WRX_b2_cnnPick/Micrographs/20171224_WRX/stack_6416_cor2_DW.mrcs'))
# STREP
# input_mrcs=torch.from_numpy(DL.fast_read_mrc('//data2/kongfang/CS-installation-test/J1/imported/018378947668571300938_stack_0495_shiny.mrcs'))

# ASE
# input_mrcs=torch.from_numpy(DL.fast_read_mrc('/data2/kongfang/data/apoe/J52/downsample/batch_000000_downsample.mrc'))


# macro
#  
# input_mrcs=torch.from_numpy(DL.fast_read_mrc('//data2/Yuanyafei/Projects/CS-lrp1/J59/downsample/batch_000000_downsample.mrc'))

#input_mrcs=torch.from_numpy(DL.fast_read_mrc('/mgh2/kongfang/BIOMAP/P73/J217/downsample/batch_000000_downsample.mrc'))

# input_mrcs=torch.from_numpy(DL.fast_read_mrc('/mgh2/kongfang/BIOMAP/P73/J236/downsample/batch_000000_downsample.mrc')).float().view(-1,1,256,256)



input_mrcs=torch.from_numpy(DL.fast_read_mrc('/mgh2/kongfang/BIOMAP/P73/J237/downsample/batch_000000_downsample.mrc')).float()


# input_mrcs_double=torch.from_numpy(DL.fast_read_mrc('/mgh2/kongfang/BIOMAP/P73/J244/downsample/batch_000000_downsample.mrc')).float()


#output=gaussian_filters(input_mrcs)

#print(output.size())
#.view(-1,256,256).float()
output=apply_lowpass(input_mrcs,steps=5).view(-1,128,128).float()
# output_double=apply_lowpass(input_mrcs_double,steps=4).view(-1,128,128).float()
mask=gen_mask(128,0.5,'2d').unsqueeze(0)
# print(mask.size())


output=torch.mul(output,mask).numpy()
# output_double=torch.mul(output_double,mask).numpy()


DL.write_file(output,'lp_forcemin128.mrc')
#DL.write_file(output_double,'lp_clean128.mrc')
