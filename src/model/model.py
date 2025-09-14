# model.py - 3D U-Net as a class
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Dropout

class UNet3D:
    def __init__(self, img_size=(128,128,128), channels=4, num_classes=4):
        self.img_size = img_size
        self.channels = channels
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        inputs = Input(shape=(*self.img_size, self.channels))
        
        # Encoder
        c1 = Conv3D(32, (3,3,3), activation='relu', padding='same')(inputs)
        c1 = Conv3D(32, (3,3,3), activation='relu', padding='same')(c1)
        p1 = MaxPooling3D((2,2,2))(c1)
        
        c2 = Conv3D(64, (3,3,3), activation='relu', padding='same')(p1)
        c2 = Conv3D(64, (3,3,3), activation='relu', padding='same')(c2)
        p2 = MaxPooling3D((2,2,2))(c2)
        
        c3 = Conv3D(128, (3,3,3), activation='relu', padding='same')(p2)
        c3 = Conv3D(128, (3,3,3), activation='relu', padding='same')(c3)
        p3 = MaxPooling3D((2,2,2))(c3)
        
        # Bottleneck
        c4 = Conv3D(256, (3,3,3), activation='relu', padding='same')(p3)
        c4 = Conv3D(256, (3,3,3), activation='relu', padding='same')(c4)
        c4 = Dropout(0.3)(c4)
        
        # Decoder
        u5 = UpSampling3D((2,2,2))(c4)
        u5 = concatenate([u5, c3])
        c5 = Conv3D(128, (3,3,3), activation='relu', padding='same')(u5)
        c5 = Conv3D(128, (3,3,3), activation='relu', padding='same')(c5)
        
        u6 = UpSampling3D((2,2,2))(c5)
        u6 = concatenate([u6, c2])
        c6 = Conv3D(64, (3,3,3), activation='relu', padding='same')(u6)
        c6 = Conv3D(64, (3,3,3), activation='relu', padding='same')(c6)
        
        u7 = UpSampling3D((2,2,2))(c6)
        u7 = concatenate([u7, c1])
        c7 = Conv3D(32, (3,3,3), activation='relu', padding='same')(u7)
        c7 = Conv3D(32, (3,3,3), activation='relu', padding='same')(c7)
        
        outputs = Conv3D(self.num_classes, (1,1,1), activation='softmax')(c7)
        
        return Model(inputs, outputs)
