from untiled import Dataloader as DL
import torch
import numpy as np


import argparse

parser=argparse.ArgumentParser('')
parser.add_argument('--ctf',action='store_true',default=False)
parser.add_argument('--filters',action='store_true',default=False)
parser.add_argument('--input_volume',type=str,default='')
parser.add_argument('--imgsize',type=int,default=256)
parser.add_argument('--b',type=int,default=20000)
parser.add_argument('--cuda',action='store_true',default=False)
parser.add_argument('--cuda_block',type=int,default=4096)
parser.add_argument('--mask',action='store_true',default=False)
parser.add_argument('--ft',action='store_true',default=False)
args = parser.parse_args()


