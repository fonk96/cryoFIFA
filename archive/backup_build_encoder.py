class Build_Encoder(nn.Module):
#   def __init__(self,backbone='resnet',basename='resnet18',rotation_dim=6,shift_dim=2,in_channels=1,activation='tanh',last_activation='tanh',image_size=64,use_batchnorm=True):
#     super(Build_Encoder,self).__init__()
#     assert backbone=='resnet' or backbone=='convnext' or backbone=='pdo' or backbone=='mlp' or backbone=='swin' or backbone=='mobile' or backbone=='efficient' or backbone=='maxvit'
#     self.flatten=False
#     self.use_batchnorm=use_batchnorm
#     if self.use_batchnorm:
#         self.batchlayers=torch.nn.BatchNorm2d(in_channels)
#     if backbone=='resnet':
#         self.feature_extractor=Res2D(basename,in_channels)
#         if basename=='resnet18' or basename=='resnet34' or basename=='resnet4':
#           self.middle_dim=512
#         else:
#           self.middle_dim=2048   
#     elif backbone=='maxvit':
#         self.feature_extractor=MaxVIT(image_size,in_channels=in_channels)
#         self.middle_dim=512     
#     elif backbone=='convnext':
#         self.feature_extractor=Convnext2D(basename,in_channels)
#         if basename=='tiny' or basename=='small':
#           self.middle_dim=768
#         elif basename=='basic':
#           self.middle_dim=1024  
#         elif basename=='large':
#           self.middle_dim=1536 
#     elif backbone=='pdo':
#         self.feature_extractor=PDO_eConvs(in_channels,image_size=image_size)
#         self.middle_dim=512   
#     elif backbone=='mlp':
#         self.flatten=True
#         self.flatten_size=in_channels*image_size*image_size
#         self.feature_extractor=Naive_MLP(self.flatten_size,256,4,256)         
#         self.middle_dim=256 
#     elif backbone=='swin':
#         if basename=='base':
#           self.middle_dim=1024
#         else:
#           self.middle_dim=768
#         self.feature_extractor=Swin(basename,in_channels=in_channels)
#     elif backbone=='mobile':
#         if basename=='v3_small':
#           self.middle_dim=576
#         else:
#           self.middle_dim=960
#         self.feature_extractor=Mobile(basename,in_channels=in_channels)
#     elif backbone=='efficient':
#         self.middle_dim=1280
#         self.feature_extractor=Efficient(basename,in_channels)

#     #self.shared_layers=Naive_MLP(self.middle_dim,256,2)



#     self.shared_layers=Naive_MLP(self.middle_dim,256,0)

#     self.rotation_layers=Naive_MLP(256,rotation_dim,0,activation=activation,last_activation=last_activation)
#     if shift_dim!=0:
#       self.shift_layers=Naive_MLP(256,shift_dim,0,activation=activation,last_activation=last_activation)
 

#     # self.rotation_layers=Naive_MLP(self.middle_dim,rotation_dim,0,activation=activation,last_activation=last_activation)
#     # if shift_dim!=0:
#     #   self.shift_layers=Naive_MLP(self.middle_dim,shift_dim,0,activation=activation,last_activation=last_activation)

#     self.shift_dim=shift_dim
#   def forward(self,x):
#     if self.use_batchnorm:
#         x=self.batchlayers(x)
#     #print(x)
#     if self.flatten :
#         x=x.view(-1,self.flatten_size)
    
#     features=self.feature_extractor(x)
#     shared_features=self.shared_layers(features)
#     # shared_features=self.feature_extractor(x)

#     rot=self.rotation_layers(shared_features)    
#     if self.shift_dim==0:
#       return rot
#     else:
#       shift=self.shift_layers(shared_features)
#       return torch.cat([rot,shift],-1)

