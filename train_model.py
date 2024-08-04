import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

def load_images(image_dir, mask_dir, img_size=(256, 256)):
    images = []
    masks = []
    
    for img_filename in os.listdir(image_dir):
        if img_filename.endswith(('.png', '.jpg', '.jpeg')):
            # Load and resize image
            img_path = os.path.join(image_dir, img_filename)
            img = Image.open(img_path).convert('RGB').resize(img_size)  # Ensure image is RGB
            img = img_to_array(img) / 255.0  # Normalize to [0, 1]
            images.append(img)
            
            # Load and resize corresponding mask
            mask_path = os.path.join(mask_dir, img_filename)
            mask = Image.open(mask_path).resize(img_size)
            mask = img_to_array(mask)
            mask = (mask > 128).astype(np.float32)  # Binarize mask
            masks.append(mask)

    return np.array(images), np.array(masks)

def build_unet(input_shape):
    inputs = Input(input_shape)
    
    # Downsampling
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    bn = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    
    # Upsampling
    u1 = UpSampling2D((2, 2))(bn)
    u1 = concatenate([u1, c4])
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(u1)
    
    u2 = UpSampling2D((2, 2))(c5)
    u2 = concatenate([u2, c3])
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u2)
    
    u3 = UpSampling2D((2, 2))(c6)
    u3 = concatenate([u3, c2])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u3)
    
    u4 = UpSampling2D((2, 2))(c7)
    u4 = concatenate([u4, c1])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c8)
    
    model = Model(inputs, outputs)
    return model

def train_model(rgb_dir, groundtruth_dir, model_path, img_size=(256, 256), epochs=50, batch_size=16):
    # Load data
    images, masks = load_images(rgb_dir, groundtruth_dir, img_size)
    
    # Build and compile the model
    model = build_unet((img_size[0], img_size[1], 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Define callbacks
    #model_path = model_path if model_path.endswith('.h5') else model_path + '.h5'
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    
    # Train the model
    model.fit(images, masks, validation_split=0.1, epochs=epochs, batch_size=batch_size, 
              callbacks=[checkpoint, early_stopping])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a U-Net model for image segmentation.")
    parser.add_argument("--rgb", required=True, help="Path to the directory containing RGB images.")
    parser.add_argument("--groundtruth", required=True, help="Path to the directory containing ground truth masks.")
    parser.add_argument("--modelpath", required=True, help="Path to save the trained model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    args = parser.parse_args()

    train_model(args.rgb, args.groundtruth, args.modelpath, epochs=args.epochs, batch_size=args.batch_size)

