import torch
import torch.nn as nn
from .Attention import SEBlock

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, relu_type='ReLU', use_bn=False, bn_type='GroupNorm', num_groups=1, is_lightweight=False) -> None:
        super(EncoderBlock, self).__init__()
        self.use_bn = use_bn
        cnn_bias = not use_bn
        inter_channels = out_channels // 2
        if is_lightweight:
            conv_groups = num_groups
        else:
            conv_groups = 1
        
        self.block1 = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=3, groups=conv_groups, stride=1, padding=1, bias=cnn_bias)
        self.block2 = nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=3, groups=conv_groups, stride=1, padding=1, bias=cnn_bias)  
        
        if relu_type == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif relu_type == 'Leaky_ReLU':
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif relu_type == 'GeLU':
            self.relu = nn.GELU()
        else:
            raise ValueError('invalid relu_type')             
        
        if self.use_bn:
            if bn_type == 'BatchNorm':
                self.bn1 = nn.BatchNorm2d(num_features=inter_channels)
                self.bn2 = nn.BatchNorm2d(num_features=inter_channels)
                self.bn3 = nn.BatchNorm2d(num_features=out_channels)
            elif bn_type == 'InstanceNorm':
                self.bn1 = nn.InstanceNorm2d(num_features=inter_channels)
                self.bn2 = nn.InstanceNorm2d(num_features=inter_channels)
                self.bn3 = nn.InstanceNorm2d(num_features=out_channels)
            elif bn_type == 'GroupNorm':
                self.bn1 = nn.GroupNorm(num_groups=num_groups, num_channels=inter_channels)
                self.bn2 = nn.GroupNorm(num_groups=num_groups, num_channels=inter_channels)
                self.bn3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            else:
                raise ValueError('invalid bn_type')
            
            self.block3 = nn.Conv2d(in_channels=inter_channels, out_channels=out_channels, kernel_size=3, groups=conv_groups, stride=1, padding=1, bias=cnn_bias)
            self.residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, groups=conv_groups, stride=1, padding=1, bias=cnn_bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bn: 
            residual = self.residual(x)       
            x = self.block1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.block2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.block3(x)
            x = self.bn3(x)            
            
        else:
            residual = x           
            x = self.block1(x)
            x = self.relu(x)
            x = self.block2(x)
            
        return x + residual    
        

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, relu_type='ReLU', use_bn=False, bn_type='GroupNorm', num_groups=2, is_lightweight=False) -> None:
        super(DecoderBlock, self).__init__()
        self.use_bn = use_bn
        cnn_bias = not use_bn
        inter_channels = in_channels
        if is_lightweight:
            conv_groups = num_groups
        else:
            conv_groups = 1
   
        self.block1 = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=3, groups=conv_groups, stride=1, padding=1, bias=cnn_bias)
        self.block2 = nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=3, groups=conv_groups, stride=1, padding=1, bias=cnn_bias)
        
        self.se = SEBlock(out_channels, relu_type)
        
        if relu_type == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif relu_type == 'Leaky_ReLU':
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif relu_type == 'GeLU':
            self.relu = nn.GELU()
        else:
            raise ValueError('invalid relu_type')             
        
        if self.use_bn:
            if bn_type == 'BatchNorm':
                self.bn1 = nn.BatchNorm2d(num_features=inter_channels)
                self.bn2 = nn.BatchNorm2d(num_features=inter_channels)
                self.bn3 = nn.BatchNorm2d(num_features=out_channels)
            elif bn_type == 'InstanceNorm':
                self.bn1 = nn.InstanceNorm2d(num_features=inter_channels)
                self.bn2 = nn.InstanceNorm2d(num_features=inter_channels)
                self.bn3 = nn.InstanceNorm2d(num_features=out_channels)
            elif bn_type == 'GroupNorm':
                self.bn1 = nn.GroupNorm(num_groups=num_groups, num_channels=inter_channels)
                self.bn2 = nn.GroupNorm(num_groups=num_groups, num_channels=inter_channels)
                self.bn3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            else:
                raise ValueError('invalid bn_type')
            
            self.block3 = nn.Conv2d(in_channels=inter_channels, out_channels=out_channels, kernel_size=3, groups=conv_groups, stride=1, padding=1, bias=cnn_bias)
            self.residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, groups=conv_groups, stride=1, padding=1, bias=cnn_bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bn: 
            residual = self.residual(x)       
            x = self.block1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.block2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.block3(x)
            x = self.bn3(x)    
            x = self.se(x)
            
        else:
            residual = x           
            x = self.block1(x)
            x = self.relu(x)
            x = self.block2(x)
            
        return x + residual   