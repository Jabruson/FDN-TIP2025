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

## Quick Start

Before running training or inference, users usually only need to check and modify the following items:

1. Dataset root in the corresponding YAML file
2. Checkpoint path in the corresponding YAML file
3. GPU number in the YAML file and shell script
4. Output directory for inference or evaluation

Recommended workflow:

1. Prepare datasets with the directory structure shown below
2. Modify `path.dataset_root` in the YAML file you want to use
3. If needed, modify pretrained checkpoint paths in `path.*`
4. Run the provided shell script or Python command directly

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

The code expects different subdirectories for different stages:

- `FDN` training uses `high_sharp` and `low_blur`
- `MAR` and `LPNet` training use `high_sharp_scaled` and `low_blur_noise`
- `LOL-v1` inference uses `lol_v1/testlow`

### Example Directory Layout

For `LOL-Blur`, a typical structure is:

```text
datasets/lolblur/
├── train/
│   ├── high_sharp/
│   ├── low_blur/
│   ├── high_sharp_scaled/
│   └── low_blur_noise/
└── test/
    ├── high_sharp/
    ├── low_blur/
    ├── high_sharp_scaled/
    └── low_blur_noise/
```

For `LOL-v1`, a typical structure is:

```text
datasets/lol_v1/
└── testlow/
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

If your dataset layout differs from the default one, you may also directly modify:

- `datasets.train.dataroot_gt`
- `datasets.train.dataroot_lq`
- `datasets.val.dataroot_gt`
- `datasets.val.dataroot_lq`
- `datasets.val_tiny.dataroot_gt`
- `datasets.val_tiny.dataroot_lq`

These paths can be relative to `dataset_root`, or you can write full absolute paths if you prefer.

### What Is `val_tiny`

`val_tiny` is only a small subset of `val`, used for faster validation during training.

- If you already prepared a tiny validation subset, point `val_tiny` to that subset
- If you do not need a separate tiny subset, just set `val_tiny` to the same paths as `val`

In other words, `val_tiny` is optional. It is a convenience setting for quick validation, not a separate required dataset.

## Which File Should Be Modified

### Training

| Stage | Command | Main YAML | What users usually need to modify |
| --- | --- | --- | --- |
| MAR | `sh MAR.sh` | `options/train/MAR_train.yml` | `path.dataset_root`, batch size, GPU count |
| FDN | `sh fdn.sh` | `options/train/FDN.yml` | `path.dataset_root`, `path.pretrain_network_mar`, batch size, GPU count |
| LPNet | `sh train_lpnet.sh` | `options/train/LPNet_train.yml` | `path.dataset_root`, batch size, GPU count |

### Inference

| Task | Command | Main YAML | What users usually need to modify |
| --- | --- | --- | --- |
| LOL-Blur | `python inference.py -opt options/test/FDN_lolblur.yml` | `options/test/FDN_lolblur.yml` | dataset root, model weights, output root |
| LOL-v1 | `python inference.py -opt options/test/FDN_lolv1.yml` | `options/test/FDN_lolv1.yml` | dataset root, model weights, output root |

## Important Fields to Modify

Below are the most important fields in the YAML files.

### 1. Dataset Root

```yaml
path:
  dataset_root: ${project_root}/datasets/lolblur
```

Modify this when your dataset is not placed under the repository `datasets/` directory.

### 2. MAR Checkpoint for FDN Training

For FDN training:

```yaml
path:
  pretrain_network_mar: ${project_root}/experiments/MAR/models
```

Modify this to:

- a specific checkpoint file, or
- a folder containing MAR checkpoints

This checkpoint is loaded into the internal `net_a` module of `FDN`.

### 4. Test-Time Checkpoints

For inference:

```yaml
path:
  pretrain_network_g: ${project_root}/checkpoint/FDN_lolblur.pth
  pretrain_network_predictor: ${project_root}/checkpoint/LPNet_lolblur.pth
```

Users should replace these paths if they want to test their own trained models.

### 5. Inference Input and Output

```yaml
inference:
  input_root: ${path.dataset_root}/test/low_blur_noise
  input_glob: ${inference.input_root}/*/*
  output_root: ${project_root}/results/FDN_lolblur
```

Modify:

- `input_root` if test images are stored elsewhere
- `input_glob` if the folder depth differs
- `output_root` if you want results saved elsewhere

### 6. GPU Settings

Please keep these consistent:

- `num_gpu` in YAML
- `--nproc_per_node` in `MAR.sh`, `fdn.sh`, and `train_lpnet.sh`

For example, if you use a single GPU, then set:

```yaml
num_gpu: 1
```

and modify the shell script to:

```bash
python -m torch.distributed.launch --nproc_per_node=1 ...
```

## Training

The main restoration pipeline is a two-stage process:

1. `MAR`: auxiliary restoration stage
2. `FDN`: main restoration network

`LPNet` is an additional illumination-ratio predictor trained separately.

The intended logic is:

1. `MAR` is trained first
2. `FDN` is trained with the ground-truth ratio and optionally initialized with MAR weights
3. `LPNet` is trained independently to predict the ratio under direct supervision from the ground-truth ratio
4. During inference/testing, `LPNet` predicts the ratio and this predicted ratio is fed into `FDN`

### Stage 1: MAR

```bash
sh MAR.sh
```

Main config: `options/train/MAR_train.yml`

Users should mainly check:

- `path.dataset_root`
- `datasets.train.dataroot_gt`
- `datasets.train.dataroot_lq`
- `num_gpu`
- `batch_size_per_gpu`

### Stage 2: FDN

```bash
sh fdn.sh
```

Main config: `options/train/FDN.yml`

Users should mainly check:

- `path.dataset_root`
- `path.pretrain_network_mar`
- `datasets.train.dataroot_gt`
- `datasets.train.dataroot_lq`
- `num_gpu`
- `batch_size_per_gpu`

### Stage 3: LPNet

```bash
sh train_lpnet.sh
```

Main config: `options/train/LPNet_train.yml`

Users should mainly check:

- `path.dataset_root`
- `datasets.train.dataroot_gt`
- `datasets.train.dataroot_lq`
- `num_gpu`
- `batch_size_per_gpu`

Training target:

- `LPNet` predicts the ratio directly
- supervision comes from the ground-truth ratio computed in the training pipeline

## Inference

We provide YAML-driven inference scripts. In most cases, you only need to edit the paths and checkpoint locations in `options/test/*.yml`.

Before inference, users should check:

- `path.dataset_root`
- `path.pretrain_network_g`
- `path.pretrain_network_predictor`
- `inference.input_root`
- `inference.input_glob`
- `inference.output_root`

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

Inference logic:

- `LPNet` first predicts a ratio from the input image
- the predicted ratio is then used as the ratio input of `FDN`

## Evaluation

### Full-reference metrics

```bash
python m.py \
  --pred_glob "results/FDN_lolblur/*/*" \
  --gt_glob "datasets/lolblur/test/high_sharp_scaled/*/*" \
  -m psnr ssim lpips \
  --output_csv results/fdn_lolblur_metrics.csv
```

Users only need to replace:

- `--pred_glob` with their own output image path
- `--gt_glob` with the corresponding ground-truth path

### FID

```bash
python fid.py \
  --pred_dir results/FDN_lolblur \
  --gt_dir datasets/lolblur/test/high_sharp_scaled
```

Users only need to replace:

- `--pred_dir`
- `--gt_dir`

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
