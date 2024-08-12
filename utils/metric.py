from cv2 import cvtColor, COLOR_BGR2GRAY
from torch import log10, Tensor
from torch.nn.functional import mse_loss
from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def get_image_metric(gt, predict) -> dict[float, float]:
    '''
    calculate image-level metrics
    return {'PSNR': float, 'SSIM': float}
    '''
    psnr = peak_signal_noise_ratio(gt, predict, data_range=255)
    if psnr == float('inf'):
        psnr = 100.0
        
    gt = cvtColor(gt, COLOR_BGR2GRAY)
    predict = cvtColor(predict, COLOR_BGR2GRAY)
    ssim = structural_similarity(predict, gt, use_sample_covariance=False, win_size=11, data_range=255)
    
    return { 'PSNR': psnr, 'SSIM': ssim }

def get_tensor_metric(gt: Tensor, predict: Tensor) -> dict[float, float]:
    '''
    calculate tensor-level metrics
    '''
    psnr_val = 10 * log10(1 / mse_loss(predict, gt)).item()
    if psnr_val == float('inf'):
        psnr_val = 100.0
    
    ssim_val = ssim(predict, gt, data_range=1.0, size_average=False).item()
    
    return { 'PSNR': psnr_val, 'SSIM': ssim_val }