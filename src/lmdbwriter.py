from new_dataloader import LmdbDataSet

import torch
import time
from untiled import Dataloader as DL
import lmdb
import numpy as np

import argparse

parser = argparse.ArgumentParser('lmdb writer')
parser.add_argument('--data_path', type=str,default='')
parser.add_argument('--particles_meta', type=str,default='')
parser.add_argument('--particles_meta_double', type=str,default='')
parser.add_argument('--imgsize',type=int,default=128)
parser.add_argument('--db_path', type=str,default='')
parser.add_argument('--double', action='store_true',default=False)
args = parser.parse_args()

particles_meta=args.particles_meta
particles_meta_double=args.particles_meta_double


data_path=args.data_path
db_path=args.db_path
imgsize=args.imgsize
skip_pose=True




stack_list,index_dict,line_index_dict,total_len,params_dict,full_pose=DL.get_full_info(particles_meta,skip_pose,pose_type=3)

if args.double:
  stack_list_add,index_dict_double,line_index_dict_double,total_len_double,params_dict_double,full_pose_double=DL.get_full_info(particles_meta_double,skip_pose,pose_type=3)	

if not args.double:
  DL.load_and_drop_particles(stack_list,index_dict,line_index_dict,total_len,prefix=data_path,db_path=db_path)
else:
  DL.load_and_drop_particles_double(stack_list,stack_list_add,index_dict,line_index_dict,total_len,prefix=data_path,db_path=db_path)
