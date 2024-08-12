import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
from .Attention import Attention

'''
Ref:
    << Vision Transformers for Single Image Dehazing >>
    IEEE TIP 2023
    DOI: 10.1109/TIP.2023.3256763
'''

class RLN(nn.Module):
	r"""Rescale LayerNorm"""
	def __init__(self, dim, eps=1e-5, detach_grad=False):
		super(RLN, self).__init__()
		self.eps = eps
		self.detach_grad = detach_grad

		self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
		self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

		self.meta1 = nn.Conv2d(1, dim, 1)
		self.meta2 = nn.Conv2d(1, dim, 1)

		trunc_normal_(self.meta1.weight, std=.02)
		nn.init.constant_(self.meta1.bias, 1)

		trunc_normal_(self.meta2.weight, std=.02)
		nn.init.constant_(self.meta2.bias, 0)

	def forward(self, input):
		mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
		std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

		normalized_input = (input - mean) / std

		if self.detach_grad:
			rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
		else:
			rescale, rebias = self.meta1(std), self.meta2(mean)

		out = normalized_input * self.weight + self.bias
		return out, rescale, rebias


class Mlp(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, relu_type='ReLU'):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features

		self.network_depth = 10
  
		if relu_type == 'ReLU':
			self.relu = nn.ReLU(inplace=True)
		elif relu_type == 'Leaky_ReLU':
			self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		elif relu_type == 'GeLU':
			self.relu = nn.GELU()
		else:
			raise ValueError('invalid relu_type')

		self.mlp = nn.Sequential(
			nn.Conv2d(in_features, hidden_features, 1),
			self.relu,
			nn.Conv2d(hidden_features, out_features, 1)
		)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.network_depth) ** (-1/4)
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		return self.mlp(x)


class TransformerBlock(nn.Module):
	def __init__(self, dim, num_heads, mlp_ratio=4., group=1,
				 norm_layer=nn.LayerNorm, mlp_norm=False,
				 window_size=8, shift_size=0, use_attn=True, relu_type='ReLU'):
		super().__init__()
		self.use_attn = use_attn
		self.mlp_norm = mlp_norm

		self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
		self.attn = Attention(dim, num_heads=num_heads, window_size=window_size,
							  shift_size=shift_size)

		self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
		self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), relu_type=relu_type)
		# self.mlp = MS_FFN(dim, relu_type=relu_type, group=group)

	def forward(self, x):
		identity = x
		if self.use_attn: x, rescale, rebias = self.norm1(x)	# RLN
		x = self.attn(x)
		if self.use_attn: x = x * rescale + rebias			# Affine
		x = identity + x

		identity = x
		if self.mlp_norm: x, rescale, rebias = self.norm2(x)
		x = self.mlp(x)
		if self.mlp_norm: x = x * rescale + rebias
		x = identity + x
		return x

class SkipFormer(nn.Module):
	def __init__(self, embed_dim, num_heads, mlp_ratio=4., norm_layer=RLN, window_size=8, group=1, relu_type='ReLU'):
		super().__init__()
		self.dim = embed_dim
		mlp_norm = False

		# build blocks
		self.block1 = TransformerBlock(dim=embed_dim, 
							 num_heads=num_heads,
							 mlp_ratio=mlp_ratio,
							 norm_layer=norm_layer,
							 mlp_norm=mlp_norm,
							 window_size=window_size,
							 shift_size=0,
							 group=group,
        					 relu_type=relu_type)

		self.block2 = TransformerBlock(dim=embed_dim, 
							 num_heads=num_heads,
							 mlp_ratio=mlp_ratio,
							 norm_layer=norm_layer,
							 mlp_norm=mlp_norm,
							 window_size=window_size,
							 shift_size=(window_size // 2),
							 group=group,
        					 relu_type=relu_type) 

	def forward(self, x):
		x = self.block1(x) 
		x = self.block2(x)   
		return x
