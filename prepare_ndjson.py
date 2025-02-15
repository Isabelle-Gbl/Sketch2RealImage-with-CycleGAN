"""
QuickDraw Sketch Preprocessing for CycleGAN

This script processes drawing data from the Google QuickDraw dataset, converting them into centered, high-contrast black-and-white images.
The processed images are intended for training a CycleGAN model.

## Requirements:
1. Download the `.ndjson` files from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified (Google QuickDraw dataset). 
   Here you can download a file from the quickdraw_dataset/full/simplified/ folder
2. Place the `.ndjson` files in a subfolder within the `root_path`. The folder name should match the desired class (e.g., `"car"` for car sketches).

## Usage:
Run the script from the command line with the following parameters:

    python prepare_ndjson.py <class_name> <image_count> --root_path <root_path>

### Parameters:
- `<class_name>`: Name of the class folder (e.g., `"car"` for car sketches).
- `<image_count>`: Number of images to process. (e.g. 4000, that is the amount of images that where used in the project.)
- `--root_path`: (Optional) Path to the dataset directory. Defaults to `Orginal_CycleGAN_Repository/datasets`.

### Example:
    python prepare_ndjson.py car 4000 --root_path /path/to/datasets

This processes 4000 car sketches from `/path/to/datasets/car/*.ndjson` and saves the images in `/path/to/datasets/car/trainA`.

## Output:
- The generated images are saved in the `trainA` subfolder of the respective class.
- The drawings are scaled, centered, contrast-enhanced, and converted to black-and-white.
- The images are ready for use as training data for a CycleGAN model.
"""

import os
import json
import argparse
from PIL import Image, ImageDraw, ImageEnhance

def load_drawing_data(path):
    """Loads drawings from an .ndjson file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found.")
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def drawing_to_image(drawing, image_size=(256, 256)):
    """Creates a centered image from a drawing."""
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    
    # Determine the bounding box of the drawing
    min_x = min(min(path[0]) for path in drawing)
    max_x = max(max(path[0]) for path in drawing)
    min_y = min(min(path[1]) for path in drawing)
    max_y = max(max(path[1]) for path in drawing)
    
    drawing_width = max_x - min_x
    drawing_height = max_y - min_y
    
    # Scale the drawing to fit the image without distortion
    scale = min((image_size[0] - 10) / drawing_width, (image_size[1] - 10) / drawing_height)
    
    # Calculate the center of the drawing
    x_center = (max_x + min_x) / 2
    y_center = (max_y + min_y) / 2
    
    for path in drawing:
        x_values = [((x - x_center) * scale + image_size[0] / 2) for x in path[0]]
        y_values = [((y - y_center) * scale + image_size[1] / 2) for y in path[1]]
        coordinates = list(zip(x_values, y_values))
        draw.line(coordinates, fill="black", width=2)
    
    # Center the drawing
    image = adjust_image_size(image, image_size)
    return image

def adjust_image_size(image, target_size=(256, 256)):
    """Scales the drawing to fit 256x256 without distortion and centers it."""
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    new_image = Image.new("RGB", target_size, "white")
    x_offset = (target_size[0] - image.size[0]) // 2
    y_offset = (target_size[1] - image.size[1]) // 2
    new_image.paste(image, (x_offset, y_offset))
    return new_image

def convert_to_black_and_white(image, threshold=128):
    """Converts the image to black and white."""
    grayscale_image = image.convert("L")
    return grayscale_image.point(lambda p: 255 if p > threshold else 0, mode="1")

def enhance_contrast(image, factor=2):
    """Enhances the contrast of the image."""
    enhancer = ImageEnhance.Contrast(image.convert("L"))
    return enhancer.enhance(factor)

def save_drawing_images(data, target_folder, num_images, image_size=(256, 256), threshold=128, contrast_factor=2):
    """Saves drawings as images."""
    os.makedirs(target_folder, exist_ok=True)
    for i, entry in enumerate(data[:num_images]):
        if "drawing" in entry:
            filename = os.path.join(target_folder, f"image_{i}.png")
            image = drawing_to_image(entry["drawing"], image_size)
            image = convert_to_black_and_white(image, threshold)
            image = enhance_contrast(image, contrast_factor)
            image.save(filename)

def process_ndjson(motif_name, image_count, root_path):
    """Processes an .ndjson file and saves the images."""
    ndjson_folder = os.path.join(root_path, motif_name)
    ndjson_files = [f for f in os.listdir(ndjson_folder) if f.endswith(".ndjson")]
    
    if not ndjson_files:
        print(f"No .ndjson file found for {motif_name}.")
        return
    
    ndjson_path = os.path.join(ndjson_folder, ndjson_files[0])
    target_folder = os.path.join(root_path, motif_name, "trainA")
    
    try:
        data = load_drawing_data(ndjson_path)
    except FileNotFoundError as e:
        print(e)
        return
    
    save_drawing_images(data, target_folder, image_count)
    print(f"Images for {motif_name} saved in {target_folder}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts drawings from an .ndjson file into images for CycleGAN.")
    parser.add_argument("motif_name", type=str, help="Name of the motif folder")
    parser.add_argument("image_count", type=int, help="Number of images to process")
    parser.add_argument("--root_path", type=str, default="Orginal_CycleGAN_Repository/datasets", help="Path to the main data directory")

    args = parser.parse_args()
    process_ndjson(args.motif_name, args.image_count, args.root_path)
