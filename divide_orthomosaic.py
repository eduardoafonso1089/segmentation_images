import os
from PIL import Image
import argparse

def divide_image(input_path, output_dir, block_size):
    # Open the input image
    img = Image.open(input_path)
    img_width, img_height = img.size

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate the number of blocks in each dimension
    x_blocks = img_width // block_size
    y_blocks = img_height // block_size

    # Iterate over the blocks
    for i in range(x_blocks):
        for j in range(y_blocks):
            left = i * block_size
            upper = j * block_size
            right = (i + 1) * block_size
            lower = (j + 1) * block_size

            # Crop the block
            box = (left, upper, right, lower)
            block = img.crop(box)

            # Save the block as an image file
            block_filename = os.path.join(output_dir, f"block_{i}_{j}.png")
            block.save(block_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide an orthomosaic image into smaller blocks.")
    parser.add_argument("--input", required=True, help="Path to the orthomosaic image file.")
    parser.add_argument("--output", required=True, help="Path to the output directory.")
    parser.add_argument("--block_size", type=int, default=256, help="Size of each block (default: 256).")
    args = parser.parse_args()

    divide_image(args.input, args.output, args.block_size)

