# Training

This repository contains a **3D U-Net implementation** for brain tumor segmentation (BraTS dataset).  
It includes **data preprocessing, model training, and optional use of Dice Loss** for better performance on imbalanced classes.

## Overview

- **Input:** 3D MRI volumes with 4 modalities (FLAIR, T1, T1ce, T2)  
- **Output:** Multi-class segmentation mask per voxel (4 classes: background, necrosis, edema, enhancing tumor)  
- **Preprocessing:**  
  - Min-Max normalization of image intensities to [0,1]  
  - Resizing to 128×128×128  
  - Mask one-hot encoding for multi-class classification  

## Loss Functions

### 1. Categorical Crossentropy (default)
\[
\text{Loss} = - \sum_{c=1}^{C} y_c \log(\hat{y}_c)
\]

- \(y_c\) = true one-hot label for class \(c\)  
- \(\hat{y}_c\) = predicted softmax probability for class \(c\)  
- Applied **voxel-wise** across the 3D volume

### 2. Dice Loss (alternative)
\[
\text{Dice} = \frac{2 |X \cap Y|}{|X| + |Y|}
\]
\[
\text{Loss} = 1 - \text{Dice}
\]

- Measures **overlap between predicted and ground truth masks**  
- Very effective for **imbalanced segmentation** (small tumors vs large background)

## Installation

```bash
pip install numpy nibabel tensorflow sklearn matplotlib
