import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
import math


def window_partition(x, window_size):
	B, H, W, C = x.shape
	x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
	windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, C)
	return windows


def window_reverse(windows, window_size, H, W):
	B = int(windows.shape[0] / (H * W / window_size / window_size))
	x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
	x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
	return x

def get_relative_positions(window_size):
	coords_h = torch.arange(window_size)
	coords_w = torch.arange(window_size)

	coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
	coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
	relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

	relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
	relative_positions_log  = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

	return relative_positions_log

    
class BorderlineBoosting(nn.Module):
    '''
    Ref: Aerial Image Dehazing with Attentive Deformable Transformers
    '''
    
    def __init__(self, dim: int, use_in_attn=False):
        super(BorderlineBoosting, self).__init__()
        
        if use_in_attn:
            self.module = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1, padding_mode='zeros'),
                nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1, output_padding=1, padding_mode='zeros'),
            )
        else:
            self.module = nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1),
                nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input - self.module(input) 
    
class SEBlock(nn.Module):
    def __init__(self, dim: int, relu_type='ReLU', reduction_ratio=4):
        super(SEBlock, self).__init__()
        # hidden_dim = max(dim // reduction_ratio, 6)
        hidden_dim = dim // 2
        
        if relu_type == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif relu_type == 'Leaky_ReLU':
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif relu_type == 'GeLU':
            self.relu = nn.GELU()
        else:
            raise ValueError('invalid relu_type')
        
        self.se = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(dim, hidden_dim, 1, bias=False),
			self.relu,
			nn.Conv2d(hidden_dim, dim, 1, bias=False),
			nn.Sigmoid(),
		)
        
    def forward(self, x):
        return self.se(x) * x


class WindowAttention(nn.Module):
	def __init__(self, dim, window_size, num_heads):

		super().__init__()
		self.dim = dim
		self.window_size = window_size  # Wh, Ww
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		relative_positions = get_relative_positions(self.window_size)
		self.register_buffer("relative_positions", relative_positions)
		self.meta = nn.Sequential(
			nn.Linear(2, 256, bias=True),
			nn.ReLU(True),
			nn.Linear(256, num_heads, bias=True)
		)

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, qkv):
		B_, N, _ = qkv.shape

		qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

		q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

		q = q * self.scale
		attn = (q @ k.transpose(-2, -1))

		relative_position_bias = self.meta(self.relative_positions)
		relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
		attn = attn + relative_position_bias.unsqueeze(0)

		attn = self.softmax(attn)

		x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
		return x


class Attention(nn.Module):
	def __init__(self, dim, num_heads, window_size, shift_size):
		super().__init__()
		self.dim = dim
		self.head_dim = int(dim // num_heads)
		self.num_heads = num_heads

		self.window_size = window_size
		self.shift_size = shift_size

		self.network_depth = 10
		# self.use_attn = use_attn
		# self.conv_type = conv_type
  
		'''
		if self.conv_type == 'Conv':
			self.conv = nn.Sequential(
				nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
				nn.ReLU(True),
				nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
			)

		if self.conv_type == 'DWConv':
			self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

		if self.conv_type == 'DWConv' or self.use_attn:
			self.V = nn.Conv2d(dim, dim, 1)
			
		'''
		self.proj = nn.Conv2d(dim, dim, 1)
		self.QK = nn.Conv2d(dim, dim * 2, 1)
		self.V = nn.Conv2d(dim, dim, 1)
		self.attn = WindowAttention(dim, window_size, num_heads)
  
		# self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')   
		self.bb = BorderlineBoosting(dim, True)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			w_shape = m.weight.shape
			
			if w_shape[0] == self.dim * 2:	# QK
				fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
				std = math.sqrt(2.0 / float(fan_in + fan_out))
				trunc_normal_(m.weight, std=std)		
			else:
				gain = (8 * self.network_depth) ** (-1/4)
				fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
				std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
				trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def check_size(self, x, shift=False):
		# Reflection padding
		_, _, h, w = x.size()
		mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
		mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

		if shift:
			x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
						  self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
		else:
			x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
		return x

	def forward(self, X):
		B, C, H, W = X.shape

		QK = self.QK(X)
		V = self.V(X)
		QKV = torch.cat([QK, V], dim=1)

		# shift
		shifted_QKV = self.check_size(QKV, self.shift_size > 0)	# reflection padding
		Ht, Wt = shifted_QKV.shape[2:]

		# partition windows
		shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
		qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

		attn_windows = self.attn(qkv)

		# merge windows
		shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

		# reverse cyclic shift (cropping)
		out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
		attn_out = out.permute(0, 3, 1, 2)

		# v_out = self.conv(V)
		v_out = self.bb(V)
		out = self.proj(v_out + attn_out)

		return out