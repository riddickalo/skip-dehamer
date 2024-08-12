import os
import random
import numpy as np
from cv2 import imread, cvtColor, resize, COLOR_BGR2RGB, COLOR_RGB2HSV_FULL, COLOR_RGB2YCR_CB
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):

    def __init__(self, root: str, scale=512, use_crop=False):
        self.use_crop = use_crop
        self.scale = scale
        self.dir_input = os.path.join(root, 'input')
        self.dir_target = os.path.join(root, 'target')
        self.input_paths = []
        self.target_paths = []
        for i in os.listdir(self.dir_input):
            self.input_paths.append(os.path.join(self.dir_input, i))

        for i in os.listdir(self.dir_target):
            self.target_paths.append(os.path.join(self.dir_target, i))

    def __getitem__(self, index):
        input_path = self.input_paths[index % len(self.input_paths)]
        target_path = self.target_paths[index % len(self.target_paths)]
        input_img = cvtColor(imread(input_path), COLOR_BGR2RGB)
        target_img = cvtColor(imread(target_path), COLOR_BGR2RGB)
        input_hsv_img = cvtColor(input_img, COLOR_RGB2HSV_FULL)
        input_ycc_img = cvtColor(input_img, COLOR_RGB2YCR_CB)

        input_img = self.transform(input_img)
        input_hsv_img = self.transform(input_hsv_img, False)
        input_ycc_img = self.transform(input_ycc_img, False)
        target_img = self.transform(target_img)
        
        if self.use_crop:
            input_img, target_img, input_hsv_img, input_ycc_img = augment_crop([input_img, target_img, input_hsv_img, input_ycc_img], size=self.scale//2)

        return {'input': input_img, 'input_HSV': input_hsv_img, 'input_YCC': input_ycc_img, 'target': target_img, 'input_paths': input_path, 'target_paths': target_path}

    def __len__(self):
        return len(self.input_paths)
    
    def transform(self, img, need_rescale=True):
        if self.scale < 512:
            img = resize(img, (self.scale, self.scale))
        img = img.astype(np.float32)
        img = img / 255.
        if need_rescale:
            return rescale(np.transpose(img, axes=[2, 0, 1]).copy())
        else:
            return np.transpose(img, axes=[2, 0, 1]).copy()
    
    
    
def rescale(img: np.ndarray) -> np.ndarray:
    return img * 2 - 1    # scale [0, 1] to [-1, 1]

def augment_crop(imgs=[], size=256, edge=0.):
    _, H, W = imgs[0].shape
    Hc, Wc = [size, size]

    if random.random() < Hc / H * edge:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H-Hc)

    if random.random() < Wc / W * edge:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W-Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][:, Hs:(Hs+Hc), Ws:(Ws+Wc)]        # C,H,W
            
    return imgs

# INFO: this is sample code of how to use the CustomDataset
if __name__ == '__main__':
    dataset = CustomDataset(root='train_dataset')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for i, data in enumerate(dataloader):
        print(data['input'].shape)  # input
        print(data['target'].shape)  # Clear
        print(data['input_paths'])
        print(data['target_paths'])
