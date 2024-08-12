# Skip-Dehamer+: Aerial Image Dehazing using a Hybrid Vision Transformer with Cross-domain Interactions
> **Abstract:**  
> In recent years, the applications of Artificial Intelligence of Things (AIoT) have been rapidly developing, with edge cameras equipped with edge computing capabilities now able to directly run deep networks on devices. Many studies have introduced deep learning techniques for dehazing aerial images. However, while existing research has considered optimizing dehazing performance using either multi-color space image information or frequency domain information, they have not utilized the advantages of both simultaneously, limiting the potential for further enhancement. To address this, our study combines the strengths of both techniques to improve dehazing performance and, for the first time, proposes Skip-Dehamer. This model employs a multi-branch encoder architecture to independently process multi-color space information. During the skip connection stage, Skip-former is used; this module is based on the image transformer architecture and introduces an borderline enhancement module to retain borderline information as much as possible between the encoding and decoding stages. Finally, in the decoding stage, the feature maps are transformed into the output dehazed images. Although considering the interactions of multi-color space image information or frequency domain information can improve the performance, it makes the model overly large and complex. Therefore, we further optimize the model by making it lightweight. Thus, we introduce Skip-Dehamer+, an enhanced version of Skip-Dehamer. Skip-Dehamer+ adds a cross-domain encoder to uniformly process multi-color space information, better extracting the interactions between domain features and improving dehazing performance. In the decoding stage, a cross-domain decoder is used, introducing a channel attention mechanism to output dehazed images. Skip-Dehamer+ is a hybrid image transformer architecture that addresses the high computational power requirements of traditional image transformers. Simultaneously, we use convolutional networks with high-pass filter characteristics and multi-head self-attention mechanisms with low-pass filter characteristics to retain high-frequency and low-frequency information in images from a frequency domain perspective, further enhancing dehazing image quality. Lastly, we employ lightweight techniques such as group convolution operations and adjusting the hidden layer dimensions of the Skip-former module, making the model more suitable for deployment on devices with computational limitations. Experimental results show that compared to the latest research, our original (Skip-Dehamer) and lightweight (Skip-Dehamer+) models improve PSNR by 6.76% and 5.13%, respectively, and SSIM by 1.14% and 1.06% on the RICE dataset. Additionally, on the Sate1K dataset, the original model shows an average PSNR improvement of 11.01% and an SSIM improvement of 0.29%, while the lightweight model shows an average PSNR improvement of 6.98%, with a slight SSIM decrease of 0.09%. In terms of model efficiency, compared to the latest research, the parameter count of the original and lightweight models decreases by 58.5% and 92.5%, respectively. In terms of FPS, our models achieve improvements of 209% and 443% compared to the latest related research using image transformers. Thus, our proposed network effectively dehazes aerial images and is deployable on edge devices.  
> **Key words: aerial image dehazing, multi-color space feature, frequency-domain enhancement, lightweight neural networks, edge computing, Internet of things**
**If you think this work is useful for your research, please cite the following:** 
```
@thesis{

}
```

## TODO
- [x] Release pre-trained model
- [x] Release code
- [ ] Submit paper
- [x] Build GitHub repository


## Network Architecture
![image](./poster/Network%20overview.png)

## Preparation
### Dependencies
- python 3.10.12
- pytorch 2.2.2 + cuda 12.2
- pipenv >= v2023.9.8

### Set Environment
```shell
# Install necessary packages
pip3 install pipenv pyenv

# Initialize a virtualenv and install dependencies from Pipfile
pipenv install          
```

## Run the code
You can modify configs in test.sh
```bash
# cd proj dir
cd skipdehamer

# testing
./run.sh --test
```