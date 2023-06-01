# Self-supervised Learning to Bring Dual Reversed Rolling Shutter Images Alive
---
[[arXiv](https://arxiv.org/abs/2305.19862)]

This repository is the official PyTorch implementation of SelfDRSC: Self-supervised Learning to Bring Dual Reversed Rolling Shutter Images Alive.

### Introduction
To correct RS distortions, existing methods adopt a fully supervised learning manner, where high framerate global shutter (GS) images should be collected as ground-truth supervision. In this paper, we propose a Self-supervised learning framework for Dual reversed RS distortions Correction (SelfDRSC), where a DRSC network can be learned to generate a high framerate GS video only based on dual RS images with reversed distortions. In particular, a bidirectional distortion warping module is proposed for reconstructing dual reversed RS images, and then a self-supervised loss can be deployed to train DRSC network by enhancing the cycle consistency between input and reconstructed dual reversed RS images. Besides start and end RS scanning time, GS images at arbitrary intermediate scanning time can also be supervised in SelfDRSC, thus enabling the learned DRSC network to generate a high framerate GS video. Moreover, a simple yet effective self-distillation strategy is introduced in self-supervised loss for mitigating boundary artifacts in generated GS images.

### Examples of the Demo
https://github.com/shangwei5/SelfDRSC/assets/43960503/6eda1861-219a-498a-899e-3c844e047ca9

https://github.com/shangwei5/SelfDRSC/assets/43960503/89eecc98-7305-4b81-99b7-432ee44b3d74


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
1.Training the first stage: (8 GPUs)
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_srsc_rsflow_multi.py --opt options/train_srsc_rsflow_multi_psnr.json --dist True
```
Please change `data_root` and `pretrained_rsg` in options according to yours.


2.Training the second stage (adding self-distillation): (8 GPUs)
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_srsc_rsflow_multi_distillv2.py --opt options/train_srsc_rsflow_multi_distillv2_psnr.json  --dist True
```
Please change `data_root`, `pretrained_rsg` and `pretrained_netG` in options according to yours.

## Cite
If you use any part of our code, or SelfDRSC is useful for your research, please consider citing:
```
@article{shang2023selfsupervised,
  title={Self-supervised Learning to Bring Dual Reversed Rolling Shutter Images Alive}, 
  author={Wei Shang and Dongwei Ren and Chaoyu Feng and Xiaotao Wang and Lei Lei and Wangmeng Zuo},
  journal={arXiv preprint arXiv:2305.19862},
  year={2023}
}
```

## Contact
If you have any questions, please contact csweishang@gmail.com.
