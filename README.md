# 🧠 3D MRI Brain Tumor Segmentation with 3D U-Net

**This repository** implements a 3D U-Net for brain tumor segmentation (BraTS-style).  
It includes preprocessing for `.nii` volumes, a custom 3D U-Net built from scratch, training utilities, and a Gradio app for interactive inference on `.npy` volumes (FLAIR, T1ce, T2 channels).

![Demo of 3D_MRI_Brain_Segmentation](MRI_segmentation_demo.mp4)

---

### 🗂️ What is a NIfTI file?  

- **NIfTI** stands for **Neuroimaging Informatics Technology Initiative**.  
- It is the **standard file format** for storing medical imaging data, especially in neuroimaging (MRI, fMRI, PET).  
- NIfTI files usually have extensions `.nii` (uncompressed) or `.nii.gz` (compressed).  
- A NIfTI file contains:  
  - **Voxel intensity values** (the actual 3D image data).  
  - **Header metadata** (voxel dimensions, orientation, patient information, scanner parameters, affine transformations).  
  - Support for **multi-dimensional data** (3D or 4D volumes — e.g., time-series MRI).  

### 🧠 Why do we use NIfTI in our case?  

- The **BraTS dataset** (used in this project) provides MRI scans in **NIfTI format** because it is the standard in medical imaging research.  
- Each `.nii` volume is a **3D array of voxels** representing brain tissue captured by MRI.  
- Unlike `.jpg` or `.png` (2D slices), `.nii` stores the **entire brain volume in 3D**, which is essential for tumor segmentation.  
- It allows us to:  
  1. **Process the full 3D structure** of tumors instead of slice-by-slice 2D approximations.  
  2. **Use multiple modalities** (FLAIR, T1, T1ce, T2) as aligned channels in the same spatial reference.  
  3. **Leverage spatial metadata** (voxel size, orientation) to ensure consistency across patients and scans.  

---

## 🔎 Project at a Glance

- **Purpose:** Segment brain tumors from multi-modal MRI scans (BraTS dataset) using a 3D U-Net.  
- **Dataset (reference):** BraTS 2020 training/validation [[Kaggle link](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)].  
- **Trained Model (example):** `brats_3d.hdf5` [[Drive link](https://drive.google.com/file/d/1p5_cGAudgRY3faVMemD79xswwSonBcoQ/view?usp=sharing)].  
- **Interactive Input:** `.npy` volumes with **3 channels** (FLAIR, T1ce, T2) for quick testing through a Gradio interface.  
- **Core Files:**
  - `segmentation_3d.ipynb` — Notebook covering data preprocessing, model training, and experiments.  
  - `src/model.py` — 3D U-Net model class implementation.  
  - `src/train.py` — Training pipeline with flexible loss functions and callbacks.

---

## ✅ Features

- **End-to-end preprocessing:** Convert NIfTI MRI volumes into normalized 3D arrays ready for network input.  
- **Custom 3D U-Net:** Encoder → bottleneck → decoder architecture with skip connections to preserve spatial detail.  
- **Flexible Loss Functions:**  
  - **Categorical Cross-Entropy (BCE):** Standard voxel-wise multi-class loss.  
  - **Dice Loss:** Measures overlap between predicted and ground-truth masks, highly effective for imbalanced tumors.  
- **Evaluation Metrics:**  
  - **Dice Coefficient (DSC):**  
    ```math
    DSC = \frac{2 |P \cap G|}{|P| + |G|}
    ```
    Measures the similarity between predicted mask \(P\) and ground truth \(G\), ranges from 0 (no overlap) to 1 (perfect overlap).  
  - **Voxel-wise Accuracy:** Percentage of correctly classified voxels across the volume.  
  - **Optional metrics:** Precision, recall, and IoU (Intersection over Union) can also be calculated per class.  
- **Gradio Interface:** Quickly visualize segmentation results on new `.npy` MRI volumes with interactive slice inspection.

---

## ⚙️ Requirements & Install

Recommended: a Linux/Mac machine with an NVIDIA GPU and >= 16GB RAM (or use reduced patch sizes).

Install dependencies:

```bash
pip install numpy nibabel tensorflow scikit-learn matplotlib gradio
