import os
import numpy as np
from PIL import Image
import argparse

def calculate_exg(image):
    """
    Calculate Excess Green Index (ExG) for vegetation detection.
    ExG = 2 * G - R - B
    """
    
    # Ensure the image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    r, g, b = image.split()
    r = np.array(r, dtype=np.float32)
    g = np.array(g, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    exg = 2 * g - r - b
    return exg

def binarize_image(input_image_path, output_image_path, threshold=0):
    # Open the input image
    img = Image.open(input_image_path)
    
    # Calculate the ExG index
    exg = calculate_exg(img)
    
    # Binarize the image based on the ExG threshold
    binary_image = np.where(exg > threshold, 1, 0).astype(np.uint8)

    # Save the binary image
    binary_image_pil = Image.fromarray(binary_image * 255)  # Convert to 0-255 range for grayscale
    binary_image_pil.save(output_image_path)

def process_directory(input_dir, output_dir, threshold=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_image_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, filename)

            binarize_image(input_image_path, output_image_path, threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binarize images based on vegetation index.")
    parser.add_argument("--input", required=True, help="Path to the directory containing input images.")
    parser.add_argument("--output", required=True, help="Path to the directory to save binarized images.")
    parser.add_argument("--threshold", type=float, default=0, help="Threshold for ExG index (default: 0).")
    args = parser.parse_args()

    process_directory(args.input, args.output, args.threshold)

