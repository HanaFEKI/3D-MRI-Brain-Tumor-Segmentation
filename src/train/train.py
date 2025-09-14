# train.py - 3D MRI Brain Tumor Segmentation with Trainer Class
import os
import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from model.model import UNet3D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

# ------------------------------
# Constants
# ------------------------------
IMG_SIZE = (128, 128, 128)
NUM_CLASSES = 4
CHANNELS = 4
BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 1e-4

# ------------------------------
# Dataset Handler
# ------------------------------
class DatasetHandler:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.scaler = MinMaxScaler()
    
    def load_nifti(self, file_path):
        return nib.load(file_path).get_fdata()
    
    def preprocess_image(self, image):
        image = self.scaler.fit_transform(image.reshape(-1,1)).reshape(image.shape)
        return np.resize(image, IMG_SIZE)
    
    def preprocess_mask(self, mask):
        mask = mask.astype(np.uint8)
        mask[mask == 4] = 3
        mask = np.resize(mask, IMG_SIZE)
        return to_categorical(mask, num_classes=NUM_CLASSES)
    
    def load_sample(self, sample_dir):
        modalities = ['flair', 't1', 't1ce', 't2']
        images = []
        for modality in modalities:
            file_path = os.path.join(sample_dir, f"{os.path.basename(sample_dir)}_{modality}.nii")
            img = self.load_nifti(file_path)
            img = self.preprocess_image(img)
            images.append(img)
        image_stack = np.stack(images, axis=-1)
        
        mask_path = os.path.join(sample_dir, f"{os.path.basename(sample_dir)}_seg.nii")
        mask = self.load_nifti(mask_path)
        mask = self.preprocess_mask(mask)
        
        return image_stack, mask
    
    def generator(self, batch_size=BATCH_SIZE):
        sample_dirs = [os.path.join(self.dataset_path, d) for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path,d))]
        while True:
            np.random.shuffle(sample_dirs)
            for i in range(0, len(sample_dirs), batch_size):
                batch_dirs = sample_dirs[i:i+batch_size]
                images, masks = [], []
                for d in batch_dirs:
                    img, mask = self.load_sample(d)
                    images.append(img)
                    masks.append(mask)
                yield np.array(images), np.array(masks)


# ------------------------------
# Dice Loss
# ------------------------------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)


# ------------------------------
# Trainer Class
# ------------------------------
class Trainer:
    def __init__(self, model, dataset_path, loss_fn='categorical_crossentropy'):
        self.model = model
        self.dataset_handler = DatasetHandler(dataset_path)
        # Choose loss function
        if loss_fn == 'dice':
            self.loss = dice_loss
        elif loss_fn == 'categorical_crossentropy':
            self.loss = 'categorical_crossentropy'
        else:
            raise ValueError("Loss function must be 'dice' or 'categorical_crossentropy'")
        self.compile_model()
    
    def compile_model(self):
        self.model.compile(
            optimizer=Adam(LEARNING_RATE),
            loss=self.loss,
            metrics=['accuracy']
        )
    
    def train(self, batch_size=BATCH_SIZE, epochs=EPOCHS):
        steps_per_epoch = max(1, len(os.listdir(self.dataset_handler.dataset_path)) // batch_size)
        checkpoint = ModelCheckpoint('brats_3d_trained.hdf5', monitor='loss', save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1)
        early_stop = EarlyStopping(monitor='loss', patience=20, verbose=1)
        
        self.model.fit(
            self.dataset_handler.generator(batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[checkpoint, reduce_lr, early_stop]
        )

# ------------------------------
# Run training
# ------------------------------
if __name__ == "__main__":
    dataset_path = "path_to_brats_dataset"  # Change this
    # Initialize model from class
    unet = UNet3D()
    
    # Pass model.model to Trainer
    trainer = Trainer(model=unet.model, dataset_path=dataset_path, loss_fn='dice')
    trainer.train()
