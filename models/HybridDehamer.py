import torch
import torch.nn as nn
import argparse
import logging
import torch.nn.functional as F
import time

from utils.common import tensor_to_numpy
from .SkipFormer import SkipFormer, PatchEmbed, PatchUnEmbed
from .ConvBlock import EncoderBlock, DecoderBlock
from .Attention import BorderlineBoosting


class HybridDehamer(nn.Module):
    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        super(HybridDehamer, self).__init__()
        self.args = args
        self.logger = logger
        self.patch_size = 4
        # dims = [16, 32, 64, 128, 64, 32, 16] 
        # dims = [32, 64, 128, 256, 128, 64, 32]    
        # dims = [20, 40, 80, 160, 80, 40, 20]    
        dims = [24, 48, 96, 192, 96, 48, 24]            
        
        # self.patch_embed = PatchEmbed(
		# 	patch_size=1, in_chans=3, embed_dim=dims[0]//2, kernel_size=3)        
        # self.patch_unembed = PatchUnEmbed(
		# 	patch_size=1, out_chans=4, embed_dim=dims[6], kernel_size=3)
        self.proj = nn.Conv2d(8, 3, 3, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # rgb
        self.encoder1 = EncoderBlock(9, dims[0], args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        self.encoder2 = EncoderBlock(dims[0], dims[1], args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        self.encoder3 = EncoderBlock(dims[1], dims[2], args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        self.encoder4 = EncoderBlock(dims[2], dims[3], args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        
        self.decoder1 =  DecoderBlock(dims[5], dims[6], args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        self.decoder1_1 =  DecoderBlock(dims[6], 8, args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        self.decoder2 =  DecoderBlock(dims[4], dims[5], args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        self.decoder2_1 =  DecoderBlock(dims[5], dims[6], args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        self.decoder3 =  DecoderBlock(dims[3], dims[4], args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        self.decoder3_1 =  DecoderBlock(dims[4], dims[5], args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        self.decoder4 =  DecoderBlock(dims[3], dims[4], args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        
        self.former1 = SkipFormer(dims[0], 2, 2., relu_type=args.relu_type)            
        self.former2 = SkipFormer(dims[1], 2, 2., relu_type=args.relu_type)
        self.former3 = SkipFormer(dims[2], 4, 4., relu_type=args.relu_type)
        self.former4 = SkipFormer(dims[3], 4, 4., relu_type=args.relu_type)
        
        self.skip1 = BorderlineBoosting(dims[0])
        self.skip2 = BorderlineBoosting(dims[1])
        self.skip3 = BorderlineBoosting(dims[2])
        
        # self.downsample1 = PatchEmbed(patch_size=2, in_chans=dims[0]//2, embed_dim=dims[1]//2)
        # self.downsample2 = PatchEmbed(patch_size=2, in_chans=dims[1]//2, embed_dim=dims[2]//2)
        # self.downsample3 = PatchEmbed(patch_size=2, in_chans=dims[2]//2, embed_dim=dims[3]//2)
        
        # self.upsample1 = PatchUnEmbed(patch_size=2, out_chans=dims[6], embed_dim=dims[5])
        # self.upsample2 = PatchUnEmbed(patch_size=2, out_chans=dims[5], embed_dim=dims[4])
        # self.upsample3 = PatchUnEmbed(patch_size=2, out_chans=dims[4], embed_dim=dims[3])
        
        self.upsample1 = nn.ConvTranspose2d(dims[6], dims[6], 2, 2)
        self.upsample2 = nn.ConvTranspose2d(dims[5], dims[5], 2, 2)
        self.upsample3 = nn.ConvTranspose2d(dims[4], dims[4], 2, 2)
              
        # hsv
        # self.patch_embed_1 = PatchEmbed(
		# 	patch_size=1, in_chans=3, embed_dim=dims[0]//2, kernel_size=3)
        
        # self.encoder1_1 = EncoderBlock(3, dims[0]//2, args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        # self.encoder2_1 = EncoderBlock(dims[0]//2, dims[1]//2, args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        # self.encoder3_1 = EncoderBlock(dims[1]//2, dims[2]//2, args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        # self.encoder4_1 = EncoderBlock(dims[2]//2, dims[3]//2, args.relu_type, args.use_bn, args.bn_type, args.num_groups)
        
        # self.downsample1_1 = PatchEmbed(patch_size=2, in_chans=dims[0]//2, embed_dim=dims[1]//2)
        # self.downsample2_1 = PatchEmbed(patch_size=2, in_chans=dims[1]//2, embed_dim=dims[2]//2)
        # self.downsample3_1 = PatchEmbed(patch_size=2, in_chans=dims[2]//2, embed_dim=dims[3]//2)
                 
    
    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
        
    def forward_features(self, rgb, mc_feats, ldg_set: bool):
        # rgb = self.patch_embed(rgb)
        # hsv = self.patch_embed_1(hsv)
        
        x = self.encoder1(torch.cat([rgb, mc_feats[0], mc_feats[1]], dim=1))            # 1
        # hsv = self.encoder1_1(hsv)
        x = self.former1(x, ldg_set)
        x1 = self.skip1(x)
        # rgb = self.downsample1(rgb)
        # hsv = self.downsample1_1(hsv)
        x = self.pool(x)
        # hsv = self.pool(hsv)
        
        x = self.encoder2(x)            # 1/2
        # hsv = self.encoder2_1(hsv)
        x = self.former2(x, ldg_set)
        x2 = self.skip2(x)
        # rgb = self.downsample2(rgb)
        # hsv = self.downsample2_1(hsv)
        x = self.pool(x)
        # hsv = self.pool(hsv)
        
        x = self.encoder3(x)            # 1/4
        # hsv = self.encoder3_1(hsv)
        x = self.former3(x, ldg_set)
        x3 = self.skip3(x)
        # rgb = self.downsample3(rgb)
        # hsv = self.downsample3_1(hsv)
        x = self.pool(x)
        # hsv = self.pool(hsv)
        
        x = self.encoder4(x)            # 1/8
        # hsv = self.encoder4_1(hsv)
        x = self.former4(x, ldg_set)
        x = self.decoder4(x)
        
        x = self.upsample3(x)        
        # x = self.fusion3([x, x3])     # 1/4
        x = self.decoder3(torch.cat([x, x3], dim=1)) 
        x = self.decoder3_1(x)  
        
        x = self.upsample2(x)
        # x = self.fusion2([x, x2])     # 1/2
        x = self.decoder2(torch.cat([x, x2], dim=1))   
        x = self.decoder2_1(x)   
        
        x = self.upsample1(x)
        # x = self.fusion1([x, x1])     # 1   
        x = self.decoder1(torch.cat([x, x1], dim=1))   
        x = self.decoder1_1(x)     
        
        # return self.patch_unembed(x)
        return self.proj(x)
        
    
    def forward(self, rgb, mc_feats=None, ldg_set=False):
        H, W = rgb.shape[2:]
        rgb = self.check_image_size(rgb)
        if mc_feats is not None:
            for i in range(len(mc_feats)):
                mc_feats[i] = self.check_image_size(mc_feats[i])
            # mc_feat = self.lgam(hsv)       # test LGAM
            feat = self.forward_features(rgb, mc_feats, ldg_set) 
        else:
            feat = self.forward_features(rgb, ldg_set)        
        
        # K, B = torch.split(feat, (1, 3), dim=1)     # soft reconstruction
        # rgb = K * rgb - B + rgb
        # rgb = rgb[:, :, :H, :W]
        return feat
    
    def infer_count_time(self, rgb, mc_feats, ldg_set: bool) -> tuple:
        start_infer = time.time()
        result = self.forward(rgb, mc_feats, ldg_set)
        finish_ifer = time.time()
        
        return tensor_to_numpy(result.clamp_(-1, 1)), finish_ifer - start_infer