# 3D-MRI-Brain-Tumor-Segmentation

This project implements a 3D U-Net model for brain tumor segmentation, initially designed for the BraTS dataset with `.nii` files, but adapted for interactive use with `.npy` files. It includes data preprocessing, a reference U-Net architecture, and an interactive Gradio interface using a pre-trained model.

## Features
- **Preprocessing**: Normalizes and resizes `.nii` data to 128x128x128 for training reference.
- **Model**: Loads a pre-trained 3D U-Net from `'brats_3d.hdf5'` for segmentation; includes a reference U-Net implementation.
- **Interface**: Gradio web app for interactive segmentation with `.npy` inputs (3 channels: FLAIR, T1ce, T2).

## Installation
Install the required packages:
```bash
pip install numpy nibabel tensorflow sklearn matplotlib gradio
