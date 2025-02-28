# 3D MRI Brain Tumor Segmentation with U-Net

This project implements a 3D U-Net model for brain tumor segmentation, designed for the BraTS dataset with `.nii` files for training reference, and adapted for interactive use with `.npy` files using a pre-trained model. It combines data preprocessing, a reference U-Net architecture, and a Gradio interface for real-time segmentation.

## Overview

- **Purpose**: Segment brain tumors from MRI scans using a pre-trained 3D U-Net model.
- **Training Reference**: Processes `.nii` files with 4 modalities (FLAIR, T1, T1ce, T2) from the BraTS dataset (see notebook).
- **Interactive Use**: Accepts `.npy` files with 3 channels (FLAIR, T1ce, T2) via a Gradio web interface.

## Files

- `segmentation_app_3d.ipynb`: Jupyter notebook with development code and detailed Markdown explanations.
- `segmentation_app_3d.py`: **Main executable Python script** for running the Gradio interface (recommended for stable execution).

## Installation

Install the required packages:
```bash
pip install numpy nibabel tensorflow sklearn matplotlib gradio
