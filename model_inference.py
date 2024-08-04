import argparse
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

def load_image(image_path, img_size=(256, 256)):
    img = Image.open(image_path).convert('RGB').resize(img_size)
    img = img_to_array(img) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img, axis=0)

def save_segmented_image(prediction, output_path):
    # Ensure the prediction is in the correct shape
    prediction = (prediction.squeeze() > 0.5).astype(np.uint8) * 255  # Binarize and scale to [0, 255]
    if len(prediction.shape) == 2:  # If the prediction is 2D, add a channel dimension
        prediction = np.expand_dims(prediction, axis=-1)
    segmented_img = array_to_img(prediction, scale=False)

    segmented_img.save(output_path + 'image.png')

def infer_model(rgb_image_path, model_path, output_path, img_size=(256, 256)):
    # Load image
    img = load_image(rgb_image_path, img_size)
    
    # Load model with error handling
    try:
        model = load_model(model_path)
    except ValueError as e:
        print(f"Error loading the model from {model_path}: {e}")
        return
    
    # Predict segmentation
    prediction = model.predict(img)
    
    # Save segmented image
    save_segmented_image(prediction, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference using a trained U-Net model for image segmentation.")
    parser.add_argument("--rgb", required=True, help="Path to the RGB image to be segmented.")
    parser.add_argument("--modelpath", required=True, help="Path to the trained model.")
    parser.add_argument("--output", required=True, help="Path to save the segmented image.")
    args = parser.parse_args()

    infer_model(args.rgb, args.modelpath, args.output)

