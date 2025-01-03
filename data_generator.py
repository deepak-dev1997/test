# data_generator.py

import os
import numpy as np
import cv2
import json
from tf_keras.utils import Sequence
from tf_keras.preprocessing.image import ImageDataGenerator,img_to_array


class DataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, target_size, num_classes, augment=True, shuffle=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.augment = augment
        self.shuffle = shuffle
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))
        self.indexes = np.arange(len(self.image_filenames))
        
        # Define augmentation parameters
        if self.augment:
            self.image_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            self.mask_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            self.image_datagen = ImageDataGenerator()
            self.mask_datagen = ImageDataGenerator()
        
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.image_filenames) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_image_filenames = [self.image_filenames[k] for k in batch_indexes]
        batch_mask_filenames = [self.mask_filenames[k] for k in batch_indexes]
        
        images = []
        masks = []
        
        for img_name, mask_name in zip(batch_image_filenames, batch_mask_filenames):
            # Load image
            img_path = os.path.join(self.image_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, self.target_size)
            img = img.astype(np.float32) / 255.0  # Normalize images
            images.append(img)
            
            # Load mask
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(np.int32)
            masks.append(mask)
        
        images = np.array(images)
        masks = np.array(masks)
        
        if self.augment:
            # Apply the same transformation to images and masks
            seed = np.random.randint(10000)
            images = self.image_datagen.flow(images, batch_size=self.batch_size, seed=seed, shuffle=False).next()
            masks = self.mask_datagen.flow(masks, batch_size=self.batch_size, seed=seed, shuffle=False).next()
        
        # One-hot encode masks
        masks = np.eye(self.num_classes)[masks]
        masks = masks.reshape((self.batch_size, self.target_size[0], self.target_size[1], self.num_classes))
        
        return images, masks
