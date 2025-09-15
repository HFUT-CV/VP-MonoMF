# VP-MonoMF: Visual Prompt guided Monocular 3D Object Detection with Multiscale Fusion

[Project Page](https://github.com/HFUT-CV/VP-MonoMF)

## 🚀 Introduction
Depth estimation from a single image is crucial for monocular 3D object detection.  
Our method **VP-MonoMF** addresses three key challenges:
- ❌ **Mutual interference** between 2D and 3D detection branches.  
- ❌ **Camera pose variance** introducing depth deviation.  
- ❌ **Poor performance on small objects** due to limited pixel area.  

We propose:
- **Multi-Depth Fusion (MDF)**: Fuses global depth maps and local 3D depth information.  
- **Two-stage training**: Train 3D detection (MDF) first, then 2D detector to avoid interference.  
- **3D Depth Reconstruction (3DR)**: Corrects depth deviation caused by camera pose.  
- **Visual Prompt Fusion (VPF)**: Enhances small object features via adaptive weighting.  

Experiments on **KITTI dataset** show VP-MonoMF achieves **state-of-the-art performance** in monocular 3D object detection.

---

## 📦 Installation
```bash
git clone https://github.com/HFUT-CV/VP-MonoMF.git
cd VP-MonoMF
conda create -n vpmonomf python=3.8
conda activate vpmonomf
pip install -r requirements.txt
