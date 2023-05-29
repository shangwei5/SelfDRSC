# Self-supervised Learning to Bring Dual Reversed Rolling Shutter Images Alive
---
[[arXiv]()]

This repository is the official PyTorch implementation of SelfDRSC: Self-supervised Learning to Bring Dual Reversed Rolling Shutter Images Alive.

### Introduction


### Examples of the Demo
![image](https://github.com/shangwei5/SelfDRSC/tree/main/video_results/r0.mp4)

### Prerequisites
- Python >= 3.8, PyTorch >= 1.7.0
- Requirements: opencv-python, numpy, matplotlib, imageio, scikit-image, tqdm


### Datasets
Please download the RS-GOPRO datasets from [link](https://drive.google.com/u/0/uc?id=1DuJphkVpvsNjgPs73y_sm4WZ8tzfxOZf&export=download).


## Download Pre-trained Model of SelfDRSC
Please download the pre-trained pwcnet from [link](https://pan.baidu.com/s/12gnAdEaJb1a_MaBuWhqPLg?pwd=pjdx)(password:pjdx). Please put these models to `./pretrained`.

Please download the pre-trained checkpoints from [Stage one](https://pan.baidu.com/s/1rEteKQfOY5St_2vfKNJC2w?pwd=42ep)(password:42ep) and [Stage two](https://pan.baidu.com/s/19EeZ38wTVjZB7pX920bIig?pwd=9w64)(password:9w64). Please put these models to `./experiments`.

## Getting Started
### 1) Testing
1.Testing on RS-GOPRO dataset:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 main_test_srsc_rsflow_multi_distillv2.py --opt options/test_srsc_rsflow_multi_distillv2_psnr.json  --dist True
```
Please change `data_root` and `pretrained_netG` in options according to yours.

1.Testing on real RS data:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 main_test_srsc_rsflow_multi_distillv2_real.py --opt options/test_srsc_rsflow_multi_distillv2_real.json  --dist True
```
Please change `data_root` and `pretrained_netG` in options according to yours.

### 2) Training
1.Training the first stage:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_srsc_rsflow_multi.py --opt options/train_srsc_rsflow_multi_psnr.json --dist True
```
Please change `data_root` and `pretrained_rsg` in options according to yours.


2.Training the second stage (adding self-distillation):
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_srsc_rsflow_multi_distillv2.py --opt options/train_srsc_rsflow_multi_distillv2_psnr.json  --dist True
```
Please change `data_root`, `pretrained_rsg` and `pretrained_netG` in options according to yours.

## Cite
If you use any part of our code, or SelfDRSC is useful for your research, please consider citing:


## Contact
If you have any questions, please contact csweishang@gmail.com.
