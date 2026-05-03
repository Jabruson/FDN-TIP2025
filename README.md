# Fourier-based Decoupling Network for Joint Low-Light Image Enhancement and Deblurring (FDN)

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-IEEE_TIP-blue.svg)](https://ieeexplore.ieee.org/document/11105001)
[![Framework](https://img.shields.io/badge/PyTorch-1.11-%23EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Official PyTorch Implementation**

**"Fourier-based Decoupling Network for Joint Low-Light Image Enhancement and Deblurring"**

**Luwei Tu, Jiawei Wu, Chenxi Wang, Deyu Meng, and Zhi Jin**

</div>

## Overview

Nighttime handheld photography is often affected by both low-light degradation and blur. FDN decouples these degradations in the Fourier domain and restores them jointly.

This repository is organized for direct reproduction and further research. The codebase now avoids hard-coded local paths in the main training, inference, and evaluation workflow. In most cases, users only need to edit the corresponding YAML file in `options/`.

## Architecture

### Overall Architecture
<div align="center">
  <img src="figs/overall.png" alt="FDN overall architecture" width="95%">
</div>

### FDformer Module
<div align="center">
  <img src="figs/fdformer.png" alt="FDformer module" width="80%">
</div>

## Installation

We recommend using Anaconda.

```bash
conda create -n fdn python=3.8
conda activate fdn

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

If you want to use a local VGG checkpoint for perceptual loss, set:

```bash
export VGG_PRETRAIN_PATH=/path/to/vgg19-dcbb9e9d.pth
```

Otherwise, torchvision will download pretrained VGG weights automatically when needed.

For convenience, we also provide a helper script:

```bash
bash scripts/download_vgg19.sh
export VGG_PRETRAIN_PATH=${PWD}/weights/vgg19-dcbb9e9d.pth
```

Note that `vgg19-dcbb9e9d.pth` is about 548 MB, so it is not suitable for a normal GitHub repository commit.

## Dataset Preparation

The LOL-Blur and Real-LOL-Blur datasets can be obtained from the [LEDNet repository](https://github.com/sczhou/LEDNet).

We recommend the following structure:

```text
FDN-TIP2025-main/
├── checkpoint/
├── datasets/
│   ├── lolblur/
│   │   ├── train/
│   │   │   ├── high_sharp/
│   │   │   ├── low_blur/
│   │   │   ├── high_sharp_scaled/
│   │   │   └── low_blur_noise/
│   │   └── test/
│   │       ├── high_sharp/
│   │       ├── low_blur/
│   │       ├── high_sharp_scaled/
│   │       └── low_blur_noise/
│   ├── lolblur_test_tinya2/
│   ├── lolblur_test_tiny2/
│   └── lol_v1/
│       └── testlow/
└── options/
```

If your dataset is stored elsewhere, edit only the dataset root in the option file:

- `options/train/MAR_train.yml`
- `options/train/FDN.yml`
- `options/train/LPNet_train.yml`
- `options/test/FDN_lolblur.yml`
- `options/test/FDN_lolv1.yml`

The key field is usually:

```yaml
path:
  dataset_root: ${project_root}/datasets/lolblur
```

## Training

The full pipeline contains three stages:

1. `MAR`: auxiliary restoration stage
2. `FDN`: main restoration network
3. `LPNet`: illumination-ratio predictor

### Stage 1: MAR

```bash
sh MAR.sh
```

Main config: `options/train/MAR_train.yml`

### Stage 2: FDN

```bash
sh fdn.sh
```

Main config: `options/train/FDN.yml`

### Stage 3: LPNet

```bash
sh train_lpnet.sh
```

Main config: `options/train/LPNet_train.yml`

### Important: Previous-Stage Checkpoint for LPNet

`LPNet` needs a pretrained FDN checkpoint. This is now controlled in the option file instead of being hard-coded inside the model:

```yaml
path:
  pretrain_network_stage1: ${project_root}/experiments/FDN/models
```

You may set `pretrain_network_stage1` to either:

- a checkpoint file, such as `experiments/FDN/models/net_g_500000.pth`
- a checkpoint directory, such as `experiments/FDN/models`

If a directory is given, the code automatically loads the latest `.pth` checkpoint in that directory.

## Inference

We provide YAML-driven inference scripts. In most cases, you only need to edit the paths and checkpoint locations in `options/test/*.yml`.

### LOL-Blur

```bash
python inference.py -opt options/test/FDN_lolblur.yml
```

or

```bash
python inference_fdn_lolblur.py
```

### LOL-v1

```bash
python inference.py -opt options/test/FDN_lolv1.yml
```

or

```bash
python inference_fdn_lolv1.py
```

Useful fields in `options/test/*.yml`:

```yaml
path:
  dataset_root: ${project_root}/datasets/lolblur
  pretrain_network_g: ${project_root}/checkpoint/FDN_lolblur.pth
  pretrain_network_predictor: ${project_root}/checkpoint/LPNet_lolblur.pth

inference:
  device: cuda:0
  input_root: ${path.dataset_root}/test/low_blur_noise
  input_glob: ${inference.input_root}/*/*
  output_root: ${project_root}/results/FDN_lolblur
```

## Evaluation

### Full-reference metrics

```bash
python m.py \
  --pred_glob "results/FDN_lolblur/*/*" \
  --gt_glob "datasets/lolblur/test/high_sharp_scaled/*/*" \
  -m psnr ssim lpips \
  --output_csv results/fdn_lolblur_metrics.csv
```

### FID

```bash
python fid.py \
  --pred_dir results/FDN_lolblur \
  --gt_dir datasets/lolblur/test/high_sharp_scaled
```

## Pretrained Models and Results

| Dataset | Google Drive | Baidu Pan |
| --- | --- | --- |
| LOL-Blur Results | [Download](https://drive.google.com/file/d/1RhGZxj0crlrEG1z4kQuuxk-yRVP9h9lI/view?usp=drive_link) | [Download](https://pan.baidu.com/s/1eDegIuW3YfuX9J9dx-T4Ig) (Code: `2khw`) |
| Real-LOL-Blur Results | [Download](https://drive.google.com/file/d/1jOaUSTRh1OYfNDYPnpHFauP_XlH21Rgv/view?usp=drive_link) | [Download](https://pan.baidu.com/s/1zibBq9YPLZ2HGXsvsmOtmA) (Code: `uw61`) |
| LOL-v1 Results | [Download](https://drive.google.com/file/d/1P-59kykpinA5MyyniBkTC8x2YT1LolU1/view?usp=drive_link) | - |

## Notes

- Please keep `num_gpu` in the YAML file consistent with `--nproc_per_node` in the shell script you use.
- The default training and testing paths are resolved from `${project_root}`.
- If your own dataset directory differs from the suggested layout, it is enough to modify the relevant `dataset_root` and checkpoint paths in the YAML file.

## Citation

```bibtex
@article{tu2025fourier,
  author={Tu, Luwei and Wu, Jiawei and Wang, Chenxi and Meng, Deyu and Jin, Zhi},
  journal={IEEE Transactions on Image Processing},
  title={Fourier-Based Decoupling Network for Joint Low-Light Image Enhancement and Deblurring},
  year={2025},
  volume={34},
  pages={5184-5199},
  doi={10.1109/TIP.2025.3592559}
}
```
