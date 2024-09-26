# Skip-Dehamer+: Aerial Image Dehazing using a Hybrid Vision Transformer with Cross-domain Interactions
**Abstract:**  
> The application of drone aerial imagery is gaining increasing attention. In existing studies on dehazing aerial images using deep learning, while multi-color space image information or frequency domain information have been considered to optimize dehazing performance, these approaches have not utilized both techniques simultaneously, limiting the potential for enhanced performance. This study combines both techniques to improve dehazing per-formance and introduces Skip-Dehamer for the first time. Skip-Dehamer includes a multi-branch encoder architecture and the Skip-former module. However, while Skip-Dehamer can im-prove performance by using multiple spatial image inputs, it also results in an overly complex model. Therefore, we further simpli-fied the model design. We introduced cross-domain encoders and decoders into Skip-Dehamer, and this improved version is called Skip-Dehamer+. The main feature of Skip-Dehamer+ is its hybrid ViT architecture, which balances the high computational power requirements of traditional ViT and optimizes from a frequency domain perspective by retaining high and low-frequency infor-mation in images, thereby enhancing the quality of dehazed im-ages. Compared to existing studies, our dehazing results show improvements of up to 17.6% in PSNR and 1.14% in SSIM, with an average performance increase of 19.2%. These results indi-cate that Skip-Dehamer+ can effectively dehaze aerial images and is suitable for deployment on edge devices.  

**Key words: aerial image dehazing, multi-color space feature, frequency-domain enhancement, lightweight neural networks, edge computing, Internet of things**

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

## TODO
- [x] Release pre-trained model
- [x] Release code
- [ ] Submit paper
- [x] Build GitHub repository
