import torch
import torch.nn as nn
import numpy as np
import os
from cv2 import imwrite, cvtColor, COLOR_RGB2BGR
import matplotlib.pyplot as plt

def prepare_folder(mode: str, prefix: str, dataset: str) -> None:
    if mode =='test':
        os.makedirs(f'checkpoints/{prefix}/test_results/{dataset}', exist_ok=True)     # just for now
        os.makedirs(f'logs/{prefix}', exist_ok=True)
    else:
        os.makedirs(f'checkpoints/{prefix}/val_results', exist_ok=True)
        os.makedirs(f'logs/{prefix}', exist_ok=True)            


def print_network(net: nn.Module, prefix: str, model_name: str) -> int:
    num_params = 0
    for params in net.parameters():
        num_params += params.numel()
        
    file_name = f'checkpoints/{prefix}/{model_name}_params.txt'
    with open(file_name, 'wt') as param_file:
        param_file.write('----------Model----------\n')
        param_file.write(net.__str__() + '\n')
        param_file.write('Total number of parameters: {0:,d}\n'.format(num_params))
        param_file.write('---------- End ----------\n')
    
    return num_params

def tensor_restore(input: torch.Tensor) -> torch.Tensor | None:
    '''
    Restore tensor from [-1, 1] to [0, 1]
    '''
    return input.mul(0.5).add(0.5)
            
 
def tensor_to_numpy(image_tensor: torch.Tensor, imtype=np.uint8):
    '''
    convert tensor to numpy array
    '''
    img_np = image_tensor.cpu().numpy()
    img_np = img_np * 255.0
    img_np = np.transpose(img_np.squeeze(0), axes=[1, 2, 0])
    return img_np.astype(imtype)
    

def save_image(image_numpy, image_path: str) -> bool:
    '''
    Save result as image, return True if save successfully.
    '''
    return imwrite(image_path, cvtColor(image_numpy, COLOR_RGB2BGR))


def plot_status(prefix: str, dataset: str, label: str, index_arr: list):
    plt.plot(index_arr, label=label)
    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.legend()
    plt.title(dataset)
    plt.savefig(f'./logs/{prefix}/{label}.png')
    plt.close()
    return
