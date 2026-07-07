import numpy as np
import argparse

parser = argparse.ArgumentParser('kcs2star')    
parser.add_argument('--cs_file', type=str,default='')

parser.add_argument('--passthrough_file', type=str,default='')

parser.add_argument('--output',type=str,default='')

args = parser.parse_args()

file_header='data_optics\n\nloop_\n_rlnVoltage #1\n_rlnSphericalAberration #2\n _rlnAmplitudeContrast #3\n_rlnImagePixelSize #4\n'
data_header='\n\ndata_particles\n\nloop_\n_rlnImageName #1 \n_rlnDefocusU #2 \n_rlnDefocusV #3 \n_rlnDefocusAngle #4\n'


optics_find_list=['ctf/accel_kv','ctf/cs_mm','ctf/amp_contrast']
data_find_list=['blob/path','blob/idx']
pass_data_find_list=['ctf/df1_A','ctf/df2_A']

data_blob=np.load(args.cs_file)
passthrouth_blob=np.load(args.passthrough_file)

data_desc=data_blob.dtype.descr
pass_desc=passthrouth_blob.dtype.descr

output_file=open(args.output,'w')

output_file.write(file_header)
output_file.flush()

optics_line=''

for item in optics_find_list:
  optics_line+=str(passthrouth_blob[0][item])+' '
optics_line+=str(data_blob[0]['blob/psize_A'])+'\n'

output_file.write(optics_line)
output_file.flush()



output_file.write(data_header)
output_file.flush()


for idx in range(len(data_blob)):
  data_line=''
  data_line+=str(int(data_blob[idx]['blob/idx'])+1)+'@'+str(data_blob[idx]['blob/path']).split('\'')[1]+' '
  for item in pass_data_find_list:

  	data_line+=str(passthrouth_blob[idx][item])+' '
  data_line+=str(float(passthrouth_blob[idx]['ctf/df_angle_rad'])/3.1415926*180.)+'\n'
  #print(data_line)
  output_file.write(data_line)
  output_file.flush()


output_file.close()

