# VP-MonoMF: Visual Prompt guided Monocular 3D Object Detection with Multiscale Fusion

[Project Page](https://github.com/HFUT-CV/VP-MonoMF)


## 📦 Installation
```bash
git clone https://github.com/HFUT-CV/VP-MonoMF.git
cd VP-MonoMF
conda create -n vpmonomf python=3.8
conda activate vpmonomf
pip install -r requirements.txt
```
## 📊 Dataset

We use the **KITTI 3D Object Detection** dataset.  
Download it from the [KITTI website](http://www.cvlibs.net/datasets/kitti/).  

Set the dataset path in `configs/dataset.yaml`:
```yaml
dataset_path: /path/to/KITTI
```
## 🔑 Training

###
Stage 1: Train the MDF module and DLA backbone.
```
python train.py --config configs/mdf.yaml
```
###
Stage 2: Train the 2D Detector and VPF module.
```
python train.py --config configs/2d.yaml
```
## 🧪 Evaluation
```
python eval.py --config configs/eval.yaml --checkpoint path/to/checkpoint.pth
```
## 📌 Citation

If you find this work useful, please cite our paper:

@article{VP-MonoMF,
  title={Visual Prompt guided Monocular 3D Object Detection with Multiscale Fusion},
  author={...},
  journal={...},
  year={2025}
}
