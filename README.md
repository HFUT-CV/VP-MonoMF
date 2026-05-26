# VP-MonoFM: Visual Prompt-guided Monocular 3D Object Detection

Monocular 3D object detection implementation with multiscale depth fusion, 3D reconstruction, and visual prompt feature enhancement.

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
    --config runs/TM_MonoVP.yaml \
    --output output/vp_monofm
```

Multi-GPU training (2 GPUs):
```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/plain_train_net.py \
    --batch_size 16 \
    --config runs/TM_MonoVP.yaml \
    --output output/vp_monofm \
    --num_gpus 2
```

## Evaluation

Evaluate a trained checkpoint:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py \
    --config runs/TM_MonoVP.yaml \
    --ckpt path/to/checkpoint.pth \
    --eval
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

Default config: `runs/TM_MonoVP.yaml`

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
