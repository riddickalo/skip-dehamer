import os
import torch
import argparse

from cv2 import imread, resize
from tqdm import tqdm
from ptflops import get_model_complexity_info
from models.SkipDehamer import SkipDehamer
from torch.utils.data import DataLoader
from utils.custom_dataset import CustomDataset
from utils.metric import get_image_metric, get_tensor_metric
from utils.logger import set_logger, close_logger, Params_logs
from utils.common import save_image, tensor_to_numpy, prepare_folder, tensor_restore

parser = argparse.ArgumentParser()
# test setting
parser.add_argument('--model_name', default='best_psnr.pth', type=str, help='model name')
parser.add_argument('--model_dir', default='best/SkipDehamer_RICE', type=str, help='path to saved models')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--model_size', default='SkipDehamer', type=str, choices=['SkipDehamer', 'SkipDehamer_light'])
parser.add_argument('--dataset', default='RICE', type=str, choices=['RICE', 'Sate1K_Thick', 'Sate1K_Moderate', 'Sate1K_Thin'], help='dataset name')
parser.add_argument('--gpu_device', default=0, type=int, help='use which GPU for training')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--metric_level', default='image', type=str, choices=['image', 'tensor'], help='choose tensor or image level metric')
parser.add_argument('--test_scale', default=256, type=int, help='testing under which image scale')
parser.add_argument('--show_flops', action='store_true', help='calcualte & show model FLOPS')

# model setting
parser.add_argument('--use_bn', default=True, type=bool, help='use batchnorm or not')
parser.add_argument('--relu_type', default='Leaky_ReLU', type=str, choices=['Leaky_ReLU', 'ReLU', 'GeLU'])
parser.add_argument('--pool_type', default='Max_Pool', type=str, choices=['Max_Pool', 'Avg_Pool'])
parser.add_argument('--bn_type', default='GroupNorm', type=str, choices=['BatchNorm', 'InstanceNorm', 'GroupNorm'])
parser.add_argument('--num_groups', type=int, default=3, help='how many groups for GroupNorm')

@torch.no_grad()
def test(args, logger, model: SkipDehamer, test_loader):
    duration = 0.0  
    PSNRs = []
    SSIMs = []  
    model.eval()
    if args.metric_level == 'image':
        for _, data in tqdm(enumerate(test_loader), desc='Testing...'):
            rgb = data['input'].cuda()
            hsv = data['input_HSV'].cuda()
            ycc = data['input_YCC'].cuda()
            
            result, t = model.infer(rgb, [hsv, ycc])
            duration += t        
            
            img_path = data['input_paths'][0]
            img_path = img_path.split('/')[-1]
            save_image(tensor_to_numpy(result), './checkpoints/' + args.model_dir + '/test_results/' + args.dataset + '/result_' + img_path)   
        
        img_list = os.listdir('./checkpoints/' + args.model_dir + '/test_results/' + args.dataset)
        for img in img_list:
            if img.startswith('result_'):
                result = imread('./checkpoints/' + args.model_dir + '/test_results/' + args.dataset + '/' + img)
                target = imread('./data/test_' + args.dataset + '/target/' + img[7:])
                
                if args.test_scale < 512:
                    target = resize(target, (args.test_scale, args.test_scale))
                
                metrics = get_image_metric(target, result)
                logger.info('{:s}: PSNR {:.4f}, SSIM {:.4f}'.format(img, metrics['PSNR'], metrics['SSIM']))
                PSNRs.append(metrics['PSNR'])
                SSIMs.append(metrics['SSIM'])   
        
        logger.info('Testing time: {:.4f}s'.format(duration))
        logger.info('Avg time: {:.4f}'.format(duration / len(img_list)))
           
    else:
        # tensor-level metrics
        for _, data in tqdm(enumerate(test_loader), desc='Compact Testing...'):
            rgb = data['input'].cuda()
            hsv = data['input_HSV'].cuda()
            ycc = data['input_YCC'].cuda()
            gt = tensor_restore(data['target'].cuda())
            
            result, t = model.infer(rgb, [hsv, ycc])
            duration += t
            
            metrics = get_tensor_metric(gt, result)
            PSNRs.append(metrics['PSNR'])
            SSIMs.append(metrics['SSIM'])

    logger.info('FPS: {:.4f}'.format(len(PSNRs) / duration))
    return sum(PSNRs) / len(PSNRs), sum(SSIMs) / len(SSIMs)


if __name__ == '__main__':
    # initialize
    args = parser.parse_args()
    
    prepare_folder('test', args.model_dir, args.dataset)    
    Params_logs('test', args)
    logger = set_logger('test', args.model_dir, 'info', True)
    logger.info('Testing Process initialized')
    
    # cuda setting
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if args.gpu_device >= torch.cuda.device_count():
            raise Exception('Invalid gpu_device number')
        else:
            torch.cuda.set_device(f'cuda:{args.gpu_device}')
            
        torch.cuda.empty_cache()
    else: 
        raise Exception('No GPU found')
    
    # data preparation
    data = CustomDataset('./data/test_' + args.dataset, args.test_scale)
    loader = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logger.info(f'Test on {args.dataset}')
    
    model = SkipDehamer(args, logger).cuda()
    logger.info('Loading model state_dicts...')
    model.load_state_dict(torch.load('./checkpoints/' + args.model_dir + '/' + args.model_name))
    psnr, ssim = test(args, logger, model, loader)
    logger.info('PSNR: {:.4f}'.format(psnr))
    logger.info('SSIM: {:.4f}'.format(ssim))
    
    if args.show_flops:
        macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=False, print_per_layer_stat=False)
        logger.info(f'FLOPs: {0.48 * macs / 1e9:.2f} GFLOPs, Param: {params / 1e6:.2f} M')  # 0.48 is the ratio refer to AIDNet test on Titan RTX
    
    close_logger()
    
