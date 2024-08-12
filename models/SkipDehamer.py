import torch
import torch.nn as nn
import argparse
import logging
import torch.nn.functional as F
import time

from utils.common import tensor_restore
from .SkipFormer import SkipFormer
from .ConvBlock import EncoderBlock, DecoderBlock
    
class SkipDehamer(nn.Module):
    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        super(SkipDehamer, self).__init__()
        self.args = args
        self.logger = logger
        self.patch_size = 4
        in_dims = 9
        out_dims = 8
        is_lightweight = False
        
        if args.model_size == 'SkipDehamer_light':
            dims = [24, 48, 96, 48, 24] 
            is_lightweight = True
        elif args.model_size == 'SkipDehamer_s':
            dims = [36, 72, 144, 72, 36]    
        else:
            dims = [48, 96, 192, 96, 48] 
        
        self.proj = nn.Conv2d(out_dims, 3, 3, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # rgb
        self.encoder1 = EncoderBlock(in_dims, dims[0], args.relu_type, args.use_bn, args.bn_type, args.num_groups, is_lightweight)
        self.encoder2 = EncoderBlock(dims[0], dims[1], args.relu_type, args.use_bn, args.bn_type, args.num_groups, is_lightweight)
        self.encoder3 = EncoderBlock(dims[1], dims[2], args.relu_type, args.use_bn, args.bn_type, args.num_groups, is_lightweight)
        
        self.decoder1 =  DecoderBlock(dims[3], dims[4], args.relu_type, args.use_bn, args.bn_type, is_lightweight=is_lightweight)
        self.decoder1_1 =  DecoderBlock(dims[4], out_dims, args.relu_type, args.use_bn, args.bn_type, is_lightweight=is_lightweight)
        self.decoder2 =  DecoderBlock(dims[2], dims[3], args.relu_type, args.use_bn, args.bn_type, is_lightweight=is_lightweight)
        self.decoder2_1 =  DecoderBlock(dims[3], dims[4], args.relu_type, args.use_bn, args.bn_type, is_lightweight=is_lightweight)
        self.decoder3 =  DecoderBlock(dims[2], dims[3], args.relu_type, args.use_bn, args.bn_type, is_lightweight=is_lightweight)
       
        self.skip1 = SkipFormer(dims[0], 2, 2., group=args.num_groups, relu_type=args.relu_type)            
        self.skip2 = SkipFormer(dims[1], 2, 2., group=args.num_groups, relu_type=args.relu_type)
        self.skip3 = SkipFormer(dims[2], 4, 4., group=args.num_groups, relu_type=args.relu_type)
        
        self.upsample1 = nn.ConvTranspose2d(dims[4], dims[4], 2, 2)
        self.upsample2 = nn.ConvTranspose2d(dims[3], dims[3], 2, 2)
                 
    
    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
        
    def forward_features(self, rgb, mc_feats):
        x = self.encoder1(torch.cat([rgb, mc_feats[0], mc_feats[1]], dim=1))
        x1 = self.skip1(x)
        x = self.pool(x)
        
        x = self.encoder2(x)
        x2 = self.skip2(x)
        x = self.pool(x)
        
        x = self.encoder3(x)
        x = self.skip3(x)
        x = self.decoder3(x) 
        
        x = self.upsample2(x)
        x = self.decoder2(torch.cat([x, x2], dim=1))   
        x = self.decoder2_1(x)   
        
        x = self.upsample1(x)
        x = self.decoder1(torch.cat([x, x1], dim=1))   
        x = self.decoder1_1(x)     
        
        return self.proj(x)
        
    def forward_flops(self, x):
        return self.forward_features(x, [x, x])       
    
    def forward(self, rgb, mc_feats=None):
        H, W = rgb.shape[2:]
        rgb = self.check_image_size(rgb)
        if mc_feats is not None:
            for i in range(len(mc_feats)):
                mc_feats[i] = self.check_image_size(mc_feats[i])
            # mc_feat = self.lgam(hsv)       # test LGAM
            feat = self.forward_features(rgb, mc_feats) 
        else:
            feat = self.forward_flops(rgb)  
        
        # K, B = torch.split(feat, (1, 3), dim=1)     # soft reconstruction
        # rgb = K * rgb - B + rgb
        # rgb = rgb[:, :, :H, :W]
        return feat + rgb
    
    def infer(self, rgb, mc_feats) -> tuple:
        '''
        Skip Dehamer inference
        return restored result and inferring time
        '''
        start_infer = time.time()
        result = self.forward(rgb, mc_feats)
        finish_ifer = time.time()
        
        return tensor_restore(result.clamp_(-1, 1)), finish_ifer - start_infer
