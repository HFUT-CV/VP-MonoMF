# VP-MonoFM: Visual Prompt-guided Monocular 3D Object Detection with Multiscale Fusion

<h5 align="center">

*Visual Prompt Feature Enhancement for Monocular 3D Object Detection*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/LICENSE)

</h5>

This repository contains an implementation of **VP-MonoFM (Visual Prompt-guided Monocular 3D Object Detection with Multiscale Fusion)**, a novel approach for monocular 3D object detection that combines:

1. **MDF (Multiscale Depth Fusion)**: Fuses global depth from 3D reconstruction and local depth from 2D-3D relationships
2. **3DR (3D Reconstruction)**: Corrects camera pose deviations using horizon and vanishing point detection  
3. **VPF (Visual Prompt Feature)**: Enhances features with learnable visual prompt masks based on object sizes

The method achieves significant improvements in depth estimation accuracy and 3D object detection performance on the KITTI benchmark.

![](figures/core.png)

## Installation

```bash
git clone https://github.com/YOUR_REPO/VP-MonoFM.git
cd VP-MonoFM

conda create -n vp_monofm python=3.8
conda activate vp_monofm

# Install PyTorch with CUDA support
conda install pytorch::pytorch torchvision torchaudio -c pytorch

# Install dependencies
pip install -r requirements.txt

# Compile DCNv2 (if using deformable convolutions)
cd model/backbone/DCNv2
sh make.sh
cd ../../..

# Install package in development mode
python setup.py develop
```

## Architecture Overview

### Two-Stage Framework

**Stage 1: Multiscale Depth Fusion (MDF)**
- Depth Detector: Estimates direct depth Zdir and reliability σdir
- Dimension Detector: Predicts 3D object dimensions (length, width, height)
- 3D Reconstruction (3DR): Corrects depth using camera pose information
- **Output**: Fused comprehensive depth Zcom combining global and local cues

**Stage 2: Visual Prompt Feature Enhancement (VPF) + 2D Detection**
- VPF Module: Enhances backbone features using learnable visual prompt masks
- 2D Detector: Estimates 2D bbox, center heatmap, offset, size, orientation, height
- **Output**: Complete 3D object detections with 2D and 3D attributes

### Key Components

| Module | Function | Input | Output |
|--------|----------|-------|--------|
| **MDF** | Depth Fusion | Backbone features | Fused depth Zcom, dimensions |
| **3DR** | Camera Correction | Depth + features | Global depth Zglo |
| **VPF** | Feature Enhancement | Features, GT boxes | Enhanced features Fvp |
| **2D Detector** | 2D Properties | Enhanced features | Center, offset, size, orientation |

## Data Preparation

Download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize as:

```
KITTI_ROOT/
├── training/
│   ├── calib/
│   ├── image_2/
│   ├── label/
│   └── ImageSets/
└── testing/
    ├── calib/
    ├── image_2/
    └── ImageSets/
```

Update `config/paths_catalog.py`:
```python
DATA_DIR = "/path/to/your/KITTI_ROOT"
```

## Training and Evaluation

### Training

Single GPU training:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py \
    --batch_size 8 \
    --config runs/monocd.yaml \
    --output output/vp_monofm
```

Multi-GPU training (2 GPUs):
```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/plain_train_net.py \
    --batch_size 16 \
    --config runs/monocd.yaml \
    --output output/vp_monofm \
    --num_gpus 2
```

### Evaluation

Evaluate a trained checkpoint:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py \
    --config runs/monocd.yaml \
    --ckpt path/to/checkpoint.pth \
    --eval
```

### Expected Performance

On KITTI val set:

| Model | AP40@Easy | AP40@Mod. | AP40@Hard | mAP |
|-------|-----------|-----------|-----------|-----|
| VP-MonoFM | TBD | TBD | TBD | TBD |

## Method Details

### Formula (1): Direct Depth Calculation
```
zi = f × dH / hi
```
Calculate depth for 4 vertical edges using focal length, 3D height, and 2D projected height.

### Formula (2)-(3): Depth Averaging
```
zd1 = (z1 + z2) / 2,  zd2 = (z3 + z4) / 2
σd1 = (σ1 + σ2) / 2,  σd2 = (σ3 + σ4) / 2
```
Average depths and reliability scores for object center estimation.

### Formula (4): Multiscale Fusion
```
Zcom = Σ(zk × σk) / Σ(σk)
where k ∈ {d1, d2, c, x'y'}
```
Weighted fusion of multiple depth estimates using reliability scores.

### Formula (5): 3D Point Conversion
```
x = (u - cu) × z / f
y = (v - cv) × z / f
```
Convert 2D pixel coordinates and depth to 3D camera coordinates.

### Formula (6)-(8): Camera Transformation
```
θR = arctan(a)
θP = arctan((xvp - cu) / f)
A = AR @ AP
```
Extract rotation angles from horizon and vanishing point, build transformation matrix.

### Formula (10): Visual Prompt Weighting
```
wi = 1 / (1 + exp((β·si - b) / T))
```
Learnable sigmoid-based weighting mechanism for object size-aware feature enhancement.

### Loss Functions (11)-(14)
```
LMDF = Ldir + Ldim + Lpose
Ldir = |Zdir - Z*| × σdir + log(1/σdir)
Ldim = Σ|dk - d*k|
Lpose = ||A - A*||F
```

## Project Structure

```
VP-MonoFM/
├── model/
│   ├── backbone/          # DLA feature extractor + DCNv2
│   ├── head/
│   │   ├── mdf_module.py               # Multiscale Depth Fusion
│   │   ├── threed_reconstruction.py    # 3D Reconstruction
│   │   ├── vpf_module.py               # Visual Prompt Feature
│   │   ├── vp_monofm_predictor.py      # Main predictor
│   │   ├── vp_monofm_loss.py           # Loss functions
│   │   └── vp_monofm_inference.py      # Post-processing
│   └── detector.py        # Main model class
├── config/                # Configuration files
├── data/                  # Data loading and augmentation
├── engine/                # Training and evaluation loops
├── tools/                 # Training scripts
├── structures/            # Data structures (boxes, instances, etc.)
└── utils/                 # Utility functions
```

## Configuration

Default config: `runs/monocd.yaml`

Key parameters in `config/defaults.py`:
```python
# Input sizes
INPUT.HEIGHT_TRAIN = 384
INPUT.WIDTH_TRAIN = 1280

# Dataset
DATASETS.MAX_OBJECTS = 40
DATASETS.DETECT_CLASSES = ("Car", "Pedestrian", "Cyclist")

# Solver
SOLVER.MAX_EPOCHS = 200
SOLVER.IMS_PER_BATCH = 8
SOLVER.BASE_LR = 1.25e-4
```
