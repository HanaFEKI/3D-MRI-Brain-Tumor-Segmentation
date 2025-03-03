# 3D MRI Brain Tumor Segmentation with U-Net
# This script implements a 3D U-Net model for brain tumor segmentation using the BraTS dataset.
# It includes data preprocessing, model training, and an interactive Gradio interface.

# Libraries
import numpy as np
import nibabel as nib
import gradio as gr
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Dropout
import threading
import random

# Constants
IMG_SIZE = (128, 128, 128)  # Target size for resizing images
NUM_CLASSES = 4  # Number of classes (0: background, 1: necrosis, 2: edema, 3: enhancing tumor)
CHANNELS = 4  # Number of MRI modalities (flair, t1, t1ce, t2)

class DatasetHandler:
    """Handles loading and preprocessing of MRI data."""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.scaler = MinMaxScaler()  # Normalizes data to [0, 1]
    
    def load_nifti(self, file_path):
        """Loads a NIfTI file and returns its data as a numpy array."""
        try:
            return nib.load(file_path).get_fdata()
        except Exception as e:
            raise ValueError(f"Failed to load NIfTI file {file_path}: {str(e)}")
    
    def preprocess_image(self, image):
        """Normalizes image data to [0, 1] range and resizes."""
        if image.size == 0:
            raise ValueError("Empty image provided for preprocessing.")
        image = self.scaler.fit_transform(image.reshape(-1, 1)).reshape(image.shape)
        return np.resize(image, IMG_SIZE)  # Resize to consistent dimensions
    
    def preprocess_mask(self, mask):
        """Preprocesses segmentation mask: converts to uint8, remaps labels, and one-hot encodes."""
        mask = mask.astype(np.uint8)
        mask[mask == 4] = 3  # Remap label 4 to 3 (BraTS convention)
        mask = np.resize(mask, IMG_SIZE)
        return to_categorical(mask, num_classes=NUM_CLASSES)  # One-hot encoding for multi-class
    
    def load_sample(self, sample_dir):
        """Loads and preprocesses a single BraTS sample with all modalities."""
        modalities = ['flair', 't1', 't1ce', 't2']
        images = []
        for modality in modalities:
            file_path = f"{sample_dir}/BraTS20_Training_{sample_dir.split('_')[-1]}_{modality}.nii"
            img = self.load_nifti(file_path)
            img = self.preprocess_image(img)
            images.append(img)
        image_stack = np.stack(images, axis=-1)  # Shape: (128, 128, 128, 4)
        
        mask_path = f"{sample_dir}/BraTS20_Training_{sample_dir.split('_')[-1]}_seg.nii"
        mask = self.load_nifti(mask_path)
        mask = self.preprocess_mask(mask)
        
        return image_stack, mask

class UNetModel:
    """Loads a pre-trained 3D U-Net model for brain tumor segmentation with reference implementation."""
    
    def __init__(self):
        self.model = self.load_pretrained_model()
    
    def load_pretrained_model(self):
        """Loads the pre-trained model from 'brats_3d.hdf5'."""
        try:
            return load_model('brats_3d.hdf5', compile=False)
        except Exception as e:
            raise ValueError(f"Failed to load pre-trained model 'brats_3d.hdf5': {str(e)}")
    
    def build_unet(self):
        """Reference implementation of the 3D U-Net architecture (not used in inference)."""
        inputs = Input((*IMG_SIZE, CHANNELS))
        
        # Encoder
        c1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        c1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c1)
        p1 = MaxPooling3D((2, 2, 2))(c1)
        
        c2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(p1)
        c2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c2)
        p2 = MaxPooling3D((2, 2, 2))(c2)
        
        c3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(p2)
        c3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c3)
        p3 = MaxPooling3D((2, 2, 2))(c3)
        
        # Bottleneck
        c4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(p3)
        c4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(c4)
        c4 = Dropout(0.3)(c4)
        
        # Decoder
        u5 = UpSampling3D((2, 2, 2))(c4)
        u5 = concatenate([u5, c3])
        c5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(u5)
        c5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c5)
        
        u6 = UpSampling3D((2, 2, 2))(c5)
        u6 = concatenate([u6, c2])
        c6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(u6)
        c6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c6)
        
        u7 = UpSampling3D((2, 2, 2))(c6)
        u7 = concatenate([u7, c1])
        c7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(u7)
        c7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c7)
        
        outputs = Conv3D(NUM_CLASSES, (1, 1, 1), activation='softmax')(c7)
        return Model(inputs=inputs, outputs=outputs)
    
    def predict(self, image):
        """Predicts segmentation mask for a given image."""
        return self.model.predict(image)

class SegmentationApp:
    """Application for interactive brain tumor segmentation."""
    
    def __init__(self, model):
        self.model = model
    
    def segment_image(self, npy_data):
        """Segments a .npy image with 3 channels and returns the result."""
        try:
            test_img_input = np.expand_dims(npy_data, axis=0)  # Add batch dimension
            prediction = self.model.predict(test_img_input)
            pred_argmax = np.argmax(prediction, axis=4)[0, :, :, :]
            
            slice_num = random.randint(0, pred_argmax.shape[2] - 1)
            
            # Display the segmented image
            fig_segmented, ax_segmented = plt.subplots(figsize=(8, 8))
            ax_segmented.imshow(pred_argmax[:, :, slice_num])
            ax_segmented.set_title("Segmentation")
            ax_segmented.axis('off')
            
            return fig_segmented, pred_argmax, npy_data[:, :, slice_num, 0]  # Return plot, prediction, and input slice
        except Exception as e:
            return None, None, None, f"Error: {str(e)}"
