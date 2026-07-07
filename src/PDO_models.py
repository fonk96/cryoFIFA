import torch
import numpy as np
from torch import nn
from torch.utils import data
import torch.nn.functional as F
import torchvision
from math import *

partial_dict_0 = torch.tensor([[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,1/2,0,0],[0,0,0,0,0],[0,0,-1/2,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,-1/4,0,1/4,0],[0,0,0,0,0],[0,1/4,0,-1/4,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,1,0,0],[0,0,-2,0,0],[0,0,1,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[-1/2,1,0,-1,1/2],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,1/2,-1,1/2,0],[0,0,0,0,0],[0,-1/2,1,-1/2,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,1,0,-1,0],[0,-1/2,0,1/2,0],[0,0,0,0,0]],
                    [[0,0,1/2,0,0],[0,0,-1,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,-1/2,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[1,-4,6,-4,1],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[-1/4,1/2,0,-1/2,1/4],[0,0,0,0,0],[1/4,-1/2,0,1/2,-1/4],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,1,-2,1,0],[0,-2,4,-2,0],[0,1,-2,1,0],[0,0,0,0,0]],
                    [[0,-1/4,0,1/4,0],[0,1/2,0,-1/2,0],[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,1/4,0,-1/4,0]],
                    [[0,0,1,0,0],[0,0,-4,0,0],[0,0,6,0,0],[0,0,-4,0,0],[0,0,1,0,0]]])


p=8
group_angle = [2*k*pi/p+pi/8 for k in range(p)]
tran_to_partial_coef_0 = [torch.tensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                     [0,cos(x),sin(x),0,0,0,0,0,0,0,0,0,0,0,0],
                                     [0,-sin(x),cos(x),0,0,0,0,0,0,0,0,0,0,0,0],
                                     [0,0,0,pow(cos(x),2),2*cos(x)*sin(x),pow(sin(x),2),0,0,0,0,0,0,0,0,0],
                                     [0,0,0,-cos(x)*sin(x),pow(cos(x),2)-pow(sin(x),2),sin(x)*cos(x),0,0,0,0,0,0,0,0,0],
                                     [0,0,0,pow(sin(x),2),-2*cos(x)*sin(x),pow(cos(x),2),0,0,0,0,0,0,0,0,0],
                                     [0,0,0,0,0,0,-pow(cos(x),2)*sin(x),pow(cos(x),3)-2*cos(x)*pow(sin(x),2),-pow(sin(x),3)+2*pow(cos(x),2)*sin(x), pow(sin(x),2)*cos(x),0,0,0,0,0],
                                     [0,0,0,0,0,0,cos(x)*pow(sin(x),2),-2*pow(cos(x),2)*sin(x)+pow(sin(x),3),pow(cos(x),3)-2*cos(x)*pow(sin(x),2),sin(x)*pow(cos(x),2),0,0,0,0,0],
                                     [0,0,0,0,0,0,0,0,0,0,pow(sin(x),2)*pow(cos(x),2),-2*pow(cos(x),3)*sin(x)+2*cos(x)*pow(sin(x),3),pow(cos(x),4)-4*pow(cos(x),2)*pow(sin(x),2)+pow(sin(x),4),-2*cos(x)*pow(sin(x),3)+2*pow(cos(x),3)*sin(x),pow(sin(x),2)*pow(cos(x),2)]]).to('cuda') for x in group_angle]

def get_coef(weight,num_inputs,num_outputs):
        #weight.size 1,7,3,3 or 56,7,3,3

        transformation = partial_dict_0[[0,1,2,3,4,5,7,8,12],1:4,1:4] #9*3*3
        transformation = transformation.view([9,9])
        transformation = transformation.to('cuda')
        #print('transformation',transformation.device)
        inv_transformation = transformation.inverse()#inverse matrix

        betas = torch.reshape(weight,(-1,9))#56*7*9
        betas = betas.to('cuda')
        betas = torch.mm(betas,inv_transformation)# 56*7*9
        betas = torch.reshape(betas,(num_inputs,num_outputs,9))

        #print('betas',betas.device)
        return betas


def z2_kernel(weight,num_inputs,num_outputs,p,partial,tran):
    og_coef = torch.reshape(weight,(num_inputs*num_outputs,9)) #(56*7)*9
    #print('og',og_coef.type())
    partial_coef = [torch.mm(og_coef,a) for a in tran]#8,(56*7)*15
    partial = torch.reshape(partial,(15,25))#15*25
    partial=partial.to('cuda')

    kernel = [torch.mm(a,partial) for a in partial_coef]#8,(56*7)*25
    kernel = torch.stack(kernel,dim=1)#(56*7)*8*25
    kernel = torch.reshape(kernel,(num_outputs*p,num_inputs,5,5))#56*56*5*5 or 56*1*5*5
    #kernel=kernel.to('cuda')
    #print('z2kernel',kernel.device)
    return kernel


import math
class open_conv2d(nn.Module):
    def __init__(self, num_inputs, num_outputs,p,partial,tran):
        super().__init__()
        self.p=p
        self.num_inputs=num_inputs
        self.num_outputs=num_outputs
        self.partial=partial
        self.tran=tran

        self.weight = nn.Parameter(torch.Tensor(self.num_inputs,self.num_outputs,3,3))
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, input):

        betas=get_coef(self.weight,self.num_inputs,self.num_outputs)

        kernel=z2_kernel(betas,self.num_inputs,self.num_outputs,self.p,self.partial,self.tran)


        input_shape = input.size()#input_size: 128,1,h,w & 128,56,h,w
        input = input.view(input_shape[0],-1, input_shape[-2], input_shape[-1])

        #y_size: 128,56,h,w

        outputs = F.conv2d(input, weight=kernel, bias=None, stride=1,
                        padding='same')

        batch_size, _, ny_out, nx_out = outputs.size()
        outputs = outputs.view(batch_size, self.num_outputs*self.p, ny_out, nx_out)
        #y_size: 128,7*8,h,w


        return outputs

class g_bn(nn.Module):
    def __init__(self,p):
        super(g_bn, self).__init__()
        self.p=p
        self.bn=nn.BatchNorm2d(7)

    def forward(self, inputs):

        channel,height,width = list(inputs.size())[1:]
        inputs = inputs.view(-1,int(channel/p),p,height,width)
        inputs = inputs.view(-1,int(channel/p),height*p,width)

        outputs=self.bn(inputs)

        outputs=outputs.view(-1,int(channel/p),p,height,width,)
        outputs = outputs.view(-1,channel,height,width)


        return outputs




class g_conv2d(nn.Module):
    def __init__(self, num_inputs, num_outputs,p,partial,tran):
        super().__init__()
        self.p=p
        self.num_inputs=int(num_inputs/p)
        self.num_outputs=num_outputs
        self.partial=partial
        self.tran=tran

        self.weight = nn.Parameter(torch.Tensor(self.p*self.num_inputs,self.num_outputs,3,3))
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, input):

        #print(self.weight.size())
        betas=get_coef(self.weight,self.num_inputs*self.p,self.num_outputs)
        og_coef = betas.view(self.num_inputs*self.p*self.num_outputs,9)#(56*7)，9

        tran_to_partial_coef = self.tran #8，9*15
        partial_coef = [torch.mm(og_coef,a) for a in tran_to_partial_coef] #8，（56*7）*15


        partial_dict = self.partial
        partial_dict = partial_dict.view(15,25)#15*25
        partial_dict = partial_dict.to('cuda')


        og_kernel_list = [torch.mm(a,partial_dict) for a in partial_coef] #8，（56*7）*25
        og_kernel_list = [og_kernel.view(self.num_inputs,self.p,self.num_outputs,25) for og_kernel in og_kernel_list] #8，（7*8*7*25）

        #print(og_kernel_list[0][:,0:,:].size(),og_kernel_list[0][:,:0,:].size())
        og_kernel_list = [torch.cat([og_kernel_list[k][:,-k:,:],og_kernel_list[k][:,:-k,:]],dim=1) for k in range(p)] #8，（7*8*7*25）


        kernel = torch.stack(og_kernel_list,dim=3)#7,8,7,8,25
        kernel = kernel.view(self.num_inputs*self.p,self.num_outputs*self.p,5,5)#56,56,5,5


        outputs = F.conv2d(input, weight=kernel, bias=None, stride=1,
                        padding='same')
        batch_size, _, ny_out, nx_out = outputs.size()
        outputs = outputs.view(batch_size, self.num_outputs*self.p, ny_out, nx_out)


        return outputs        


class P8_PDO_Conv_Z2(open_conv2d):

    def __init__(self, num_inputs,*args, **kwargs):
        super(P8_PDO_Conv_Z2, self).__init__(num_inputs=num_inputs, num_outputs=7,p=8,partial=partial_dict_0,tran=tran_to_partial_coef_0)

class P8_PDO_Conv_P8(g_conv2d):

    def __init__(self, *args, **kwargs):
        super(P8_PDO_Conv_P8, self).__init__(num_inputs=56, num_outputs=7,p=8,partial=partial_dict_0,tran=tran_to_partial_coef_0)
class BN_P8(g_bn):

    def __init__(self, *args, **kwargs):
        super(BN_P8, self).__init__(p=8)

class PDO_eConvs(nn.Module):
    def __init__(self,in_channels=1,image_size=64):
        super(PDO_eConvs, self).__init__()

        self.in_channels=in_channels
        self.image_size=image_size

 
        layers=[]
        layers.append(P8_PDO_Conv_Z2(in_channels,7,8))
        layers.append(nn.Tanh())
        layers.append( BN_P8(8))      


        layers.append(P8_PDO_Conv_P8(56,7,8))
        layers.append(nn.Tanh())
        layers.append( BN_P8(8))    
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))        


        for i in range(4):
            layers.append(P8_PDO_Conv_P8(56,7,8))
            layers.append(nn.Tanh())
            layers.append( BN_P8(8))
            # layers.append(nn.Dropout(p=0.2))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) 

        final_layer_size=int(56*(image_size/32)**2)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(final_layer_size, 512))
        layers.append(nn.Tanh())
        self.layers=nn.Sequential(*layers)



    def forward(self, x):
        #x=x.view(-1,self.in_channels,self.image_size*self.image_size)
        return self.layers(x)


