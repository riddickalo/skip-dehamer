import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F
from collections import OrderedDict
from math import exp
from utils.common import tensor_restore

'''
    Total loss = L1/Charbonnier loss + Borderline Edge loss
'''

class TotalLoss(nn.Module):
    def __init__(self, use_l1: bool, use_edge: bool, use_percept: bool, use_msssim: bool,
                 lambda_l1: float, lambda_edge: float, lambda_percept: float, lambda_ssim: float,
                 use_rescale: bool):
        
        super(TotalLoss, self).__init__()
        self.use_rescale = use_rescale
        self.use_l1 = use_l1
        self.use_edge = use_edge
        self.use_percept = use_percept
        self.use_msssim = use_msssim
        
        if use_percept:
            self.loss_netowrk = PercetualLoss()
        if use_edge:
            self.edge_loss = EdgeLoss()
            
        self.Charbonnier = Charbonnier_loss()
        
        self.lambda_edge = lambda_edge
        self.lambda_l1 = lambda_l1
        self.lambda_percept = lambda_percept     
        self.lambda_ssim = lambda_ssim 
        
    def cal(self, gt, predict):
        loss_trace = OrderedDict()
        total_loss = 0.0
        if self.use_percept:
            loss_trace['perceptual_loss'] = self.lambda_percept * self.loss_netowrk(predict, gt) 
            total_loss += loss_trace['perceptual_loss']
        if self.use_msssim: 
            loss_trace['ms_ssim_loss'] = self.lambda_ssim * ms_ssim_loss(predict, gt, rescale=self.use_rescale)
            total_loss += loss_trace['ms_ssim_loss']
        if self.use_edge:
            loss_trace['edge_loss'] = self.lambda_edge * self.edge_loss(predict, gt)
            total_loss += loss_trace['edge_loss']
        if self.use_l1:
            loss_trace['l1_loss'] = self.lambda_l1 * F.l1_loss(predict, gt) 
            # loss_trace['l1_loss'] = self.lambda_l1 * self.Charbonnier(predict, gt) 
            # loss_trace['l1_loss'] = self.lambda_l1 * F.smooth_l1_loss(predict, gt) 
            total_loss += loss_trace['l1_loss']
        
        loss_trace['total_loss'] = total_loss
        return total_loss, loss_trace
    
    def trace_msg(self, trace: OrderedDict) -> str:
        msg = ''
        if self.use_percept:
            msg += f"perceptual: {torch.mean(trace['perceptual_loss'])}, "
        if self.use_msssim:
            msg += f"ms-ssim: {torch.mean(trace['ms_ssim_loss'])}, "
        if self.use_edge:
            msg += f"edge: {torch.mean(trace['edge_loss'])}, "
        if self.use_l1:
            msg += f"L1: {torch.mean(trace['l1_loss'])}, "
        
        return msg
                        
        

class PercetualLoss(nn.Module):
    def __init__(self):
        super(PercetualLoss, self).__init__()
        self.mse = nn.MSELoss()
        vgg = vgg16(pretrained=True).eval()
        self.loss_layer3 = nn.Sequential(*list(vgg.features)[:3]).cuda().eval()
        self.loss_layer8 = nn.Sequential(*list(vgg.features)[:8]).cuda().eval()
        self.loss_layer15 = nn.Sequential(*list(vgg.features)[:15]).cuda().eval()
    
    def forward(self, x, y):
        loss3 = self.mse(self.loss_layer3(x), self.loss_layer3(y))
        loss8 = self.mse(self.loss_layer8(x), self.loss_layer8(y))
        loss15 = self.mse(self.loss_layer15(x), self.loss_layer15(y))
        
        return (loss3 + loss8 + loss15) / 3
    
class EdgeLoss(nn.Module):
    def __init__(self, num_gpu=0) -> None:
        super(EdgeLoss, self).__init__()
        # self.device = torch.device('cuda', num_gpu)

    def rgb_to_grayscale(self, img):
        r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.unsqueeze(1)

    def forward(self, result: torch.Tensor, target: torch.Tensor):
        result = self.rgb_to_grayscale(result).cuda()
        target = self.rgb_to_grayscale(target).cuda()
        weight = torch.FloatTensor([-1, -1, -1, -1, 8, -1, -1, -1, -1]).view(1, 1, 3, 3).cuda()
        result_edge = F.conv2d(result, weight=weight, padding=1)
        target_edge = F.conv2d(target, weight=weight, padding=1)
        loss = F.mse_loss(result_edge, target_edge)

        return torch.mean(loss)
    
'''
Charbonnier loss
'''
    
class Charbonnier_loss(nn.Module):
    """L1 Charbonnier loss."""
    def __init__(self, eps=1e-6):
        super(Charbonnier_loss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


'''
MS-SSIM Loss
'''
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).cuda()

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret
    
def ms_ssim_loss(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False, rescale=False):
    weights =torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda() # multi-scale factors
    levels = weights.size()[0]
    msssim = []
    mcs = []
    if rescale:
        img1 = tensor_restore(img1)
        img2 = tensor_restore(img2)
    
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        msssim.append(sim)
        mcs.append(cs)
        
        img1 = F.avg_pool2d(img1, (2,2))
        img2 = F.avg_pool2d(img2, (2,2))
        
    msssim = torch.stack(msssim)
    mcs = torch.stack(mcs)
    
    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        msssim = (msssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = msssim ** weights
    output = 1 - torch.prod(pow1[:-1] * pow2[-1])
    return output