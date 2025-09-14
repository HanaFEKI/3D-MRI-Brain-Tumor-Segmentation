# 🧠 3D MRI Brain Tumor Segmentation with 3D U-Net

**This repository** implements a 3D U-Net for brain tumor segmentation (BraTS-style).  
It includes preprocessing for `.nii` volumes, a custom 3D U-Net built from scratch, training utilities, and a Gradio app for interactive inference on `.npy` volumes (FLAIR, T1ce, T2 channels).

---

## 🔎 Project at a glance

- **Purpose:** Segment brain tumors from multi-modal MRI (BraTS).  
- **Dataset (reference):** BraTS training/validation [[Kaggle link](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)].  
- **Trained model (example):** `brats_3d.hdf5` [[Drive link]](https://drive.google.com/file/d/1p5_cGAudgRY3faVMemD79xswwSonBcoQ/view?usp=sharing).  
- **Interactive input:** `.npy` volumes with **3 channels** (FLAIR, T1ce, T2) for the Gradio demo.  
- **Core files:**
  - `segmentation_app_3d.ipynb` — notebook with development, preprocessing pipeline, and experiments.
  - `segmentation_app_3d.py` — recommended stable entrypoint (Gradio server + inference helpers).
  - `train.py` — training script for 3D U-Net on `.nii` (BraTS) data.

---

## ✅ Features

- End-to-end preprocessing (NIfTI → normalized 3D arrays).
- Fully custom 3D U-Net implementation (encoder / bottleneck / decoder with skip connections).
- Losses & metrics suitable for medical segmentation (Dice loss, BCE, Dice coefficient).
- Training utilities: augmentation, patch extraction, checkpointing.
- Gradio interface for quick model inspection and demo.

---

## ⚙️ Requirements & Install

Recommended: a Linux/Mac machine with an NVIDIA GPU and >= 16GB RAM (or use reduced patch sizes).

Install dependencies:

```bash
pip install numpy nibabel tensorflow scikit-learn matplotlib gradio
# optional: albumentations, torch (if you adapt to PyTorch), tqdm
