# train.py

import matplotlib.pyplot as plt
import numpy as np 
import os
import tensorflow as tf
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_generator import DataGenerator  # Import the custom data generator
from model2 import unet
from tf_keras.preprocessing.image import ImageDataGenerator, img_to_array
from tf_keras.utils import to_categorical
from tf_keras.losses import categorical_crossentropy
from tf_keras import backend as K
from PIL import Image
from tf_keras.metrics import MeanIoU
from tf_keras.optimizers import Adam

# Define Dice and Combined Loss Functions
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def get_num_classes(mask_folder):
    max_label = 0
    for root, _, files in os.walk(mask_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                mask_path = os.path.join(root, file)
                with Image.open(mask_path) as img:
                    # Ensure the mask is in grayscale
                    mask = np.array(img.convert('L')).astype(np.int32)
                current_max = mask.max()
                print(f"Processing {mask_path}: max label = {current_max}")
                if current_max > max_label:
                    max_label = current_max
    print(f"Final max label: {max_label}")
    return max_label + 1

# Define paths
train_path = 'train2'  # Base directory containing 'images' and 'labels' folders
image_folder = 'trainimages'
mask_folder = 'maskedimages'
saved_model_path = os.path.join('saved_models2', 'unet_rooftop.h5')

aug_dict = dict(
    rotation_range=20,           # Degrees (0 to 180)
    width_shift_range=0.05,      # Fraction of total width
    height_shift_range=0.05,     # Fraction of total height
    shear_range=0.05,            # Shear angle in counter-clockwise direction in degrees
    zoom_range=0.05,             # Range for random zoom
    horizontal_flip=True,        # Randomly flip inputs horizontally
    fill_mode='nearest'          # Points outside the boundaries are filled
)

batch_size = 2
target_size = (256, 256)
seed = 1
num_classes = get_num_classes(mask_folder=os.path.join(train_path, 'maskedimages'))  # Updated from 9 to 10
print(f"Number of classes: {num_classes}")

# Create ImageDataGenerators
image_datagen = ImageDataGenerator(**aug_dict, rescale=1./255)
mask_datagen = ImageDataGenerator(**aug_dict)

# Ensure that the seed is the same for image and mask generators
image_generator = image_datagen.flow_from_directory(
    train_path,
    classes=[image_folder],
    class_mode=None,
    color_mode='rgba',                # Adjust based on your image channels
    target_size=target_size,
    batch_size=batch_size,
    seed=seed
)

mask_generator = mask_datagen.flow_from_directory(
    train_path,
    classes=[mask_folder],
    class_mode=None,
    color_mode='grayscale',
    target_size=target_size,
    batch_size=batch_size,
    seed=seed
)

# Combine generators into one
def combine_generators(image_gen, mask_gen):
    while True:
        img = next(image_gen)
        mask = next(mask_gen)
        yield (img, mask)

train_generator = combine_generators(image_generator, mask_generator)

# Custom generator to process masks
def multi_class_generator(generator, num_classes):
    for (img, mask) in generator:
        # Masks are integer encoded; ensure they are integers
        mask = mask[..., 0]  # Remove the channel dimension
        mask = mask.astype(np.int32)
        
        current_max = mask.max()
        # print(f"Current batch max label: {current_max}")
        # Optional: Verify that mask labels are within the valid range
        if mask.max() >= num_classes:
            raise ValueError(f"Mask contains labels >= num_classes ({num_classes}).")
        
        # Convert masks to categorical (one-hot encoding)
        mask = to_categorical(mask, num_classes=num_classes)
        yield (img, mask)

myGene = multi_class_generator(train_generator, num_classes)

# Visualize some augmented images and masks
def visualize_generator(generator, batch_size=2, num_classes=10):
    example = next(generator)
    images, masks = example
    for i in range(batch_size):
        img = images[i]
        mask = masks[i]
        mask = np.argmax(mask, axis=-1)  # Convert one-hot back to single channel
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[0].set_title('Augmented Image')
        ax[1].imshow(mask, cmap='jet', vmin=0, vmax=num_classes-1)
        ax[1].set_title('Augmented Mask')
        plt.show()

# Uncomment to visualize
# visualize_generator(myGene, batch_size, num_classes)

# Initialize and compile model
model = unet(input_size=(256, 256, 4), num_classes=num_classes)

model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss=combined_loss, 
              metrics=['accuracy', MeanIoU(num_classes=num_classes)])

model.summary()

# Define training parameters
epochs = 50
steps_per_epoch = 500  # Adjust based on dataset size

# Add callbacks for better training control
callbacks = [
    EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True),
    ModelCheckpoint(saved_model_path, monitor='loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)
]

# Train the model
history = model.fit(
    myGene,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=callbacks
)

# Save the trained model (if not using ModelCheckpoint)
os.makedirs('saved_models2', exist_ok=True)
model.save(saved_model_path)
print(f"Model saved to {saved_model_path}")
