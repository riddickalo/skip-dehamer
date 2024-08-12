import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from .Attention import BorderlineBoosting

'''
Ref:
    << SCANet: Self-Paced Semi-Curricular Attention Network for Non-Homogeneous Image Dehazing >>
    CVPRW 2023
    arXiv:2304.08444
'''

class ChannelAttention(nn.Module):
    def __init__(self, nc,number, norm_layer = nn.BatchNorm2d):
        super(ChannelAttention, self).__init__()
        self.conv1 = nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True)
        self.bn1 = norm_layer(nc)
        self.prelu = nn.PReLU(nc)
        self.conv2 = nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True)
        self.bn2 = norm_layer(nc)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(nc, number, 1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(number, nc, 1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        se = self.gap(x)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return se
    
class SpatialAttention(nn.Module):
    def __init__(self, nc, number, norm_layer = nn.BatchNorm2d):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(nc,nc,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = norm_layer(nc)
        self.prelu = nn.PReLU(nc)
        self.conv2 = nn.Conv2d(nc,number,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = norm_layer(number)
        
        self.conv3 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=3,dilation=3,bias=False)
        self.conv4 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=5,dilation=5,bias=False)
        self.conv5 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=7,dilation=7,bias=False)
        
        self.fc1 = nn.Conv2d(number*4,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        x1 = x
        x2 = self.conv3(x)
        x3 = self.conv4(x)
        x4 = self.conv5(x)
        
        se = torch.cat([x1, x2, x3, x4], dim=1)
        
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        
        return se

class _residual_block_ca(nn.Module):
    def __init__(self,nc, number = 4, norm_layer = nn.BatchNorm2d):
        super(_residual_block_ca,self).__init__()
        self.CA = ChannelAttention(nc,number)
        self.MSSA = SpatialAttention(nc,number)
    def forward(self,x):
        x0 = x
        x1 = self.CA(x)*x
        x2 = self.MSSA(x1)*x1
        
        return x0+x2

class LGAM(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, in_features=32, n_residual_att=3):
        super(LGAM, self).__init__()
        
         # ver.1
        att = [ nn.ReflectionPad2d(2),
	            nn.Conv2d(in_channels, in_features//2, 3),
                # nn.BatchNorm2d(in_features//2),
                nn.GroupNorm(num_groups=2, num_channels=in_features//2),
                nn.PReLU() ]
        for _ in range(n_residual_att):
            att += [ _residual_block_ca(in_features//2) ]
    
        att += [ #nn.ReflectionPad2d(1),
                 nn.Conv2d(in_features//2, out_channels, 3),
                #  nn.Sigmoid() 
                ]
    
        self.att = nn.Sequential(*att)
        
        # ver.2        
        # self.extractor = nn.Sequential(
        #                     nn.Conv2d(in_channels, in_features, kernel_size=3, stride=1, padding=1),
        #                     nn.BatchNorm2d(num_features=in_features),
        #                     nn.PReLU(),
        #                     nn.Conv2d(in_features, in_channels, kernel_size=3, stride=1, padding=1),
        #                     nn.BatchNorm2d(num_features=in_channels),
        #                     nn.PReLU(),
        #                     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        #                     nn.BatchNorm2d(num_features=in_channels),
        #                     # nn.PReLU(),
        #                 )        
        # self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)     
        
    def forward(self, x):        
        return self.att(x)
        
        # residual = x        
        # x = self.extractor(x)        
        # return self.proj(x + residual)