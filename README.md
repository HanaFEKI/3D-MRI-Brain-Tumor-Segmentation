# 3D MRI Brain Tumor Segmentation with U-Net

This project implements a 3D U-Net model for brain tumor segmentation, designed for the BraTS dataset with .nii files for training reference, and adapted for interactive use with .npy files using a pre-trained model. It combines data preprocessing, a custom U-Net architecture built from scratch, and a Gradio interface for real-time segmentation.

## Overview

- **Purpose**: Segment brain tumors from MRI scans using a custom-trained 3D U-Net model.
- **Dataset**: The BraTS dataset can be accessed at: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation.
- **Training Reference**: Processes .nii files with 4 modalities (FLAIR, T1, T1ce, T2) from the BraTS dataset (see notebook).
- **Interactive Use**: Accepts .npy files with 3 channels (FLAIR, T1ce, T2) via a Gradio web interface.
- **Trained Model**: The trained model `brats_3d.hdf5` is available in the following Drive link: https://drive.google.com/file/d/1p5_cGAudgRY3faVMemD79xswwSonBcoQ/view?usp=sharing.
  
## Files

- `segmentation_app_3d.ipynb`: Jupyter notebook with development code and detailed Markdown explanations.
- `segmentation_app_3d.py`: **Main executable Python script** for running the Gradio interface (recommended for stable execution).
- `train.py`: Script for training the 3D U-Net model on the BraTS dataset.

## Installation

Install the required packages:

```bash
pip install numpy nibabel tensorflow sklearn matplotlib gradio
```

```bash
pip install numpy nibabel tensorflow sklearn matplotlib gradio
```



