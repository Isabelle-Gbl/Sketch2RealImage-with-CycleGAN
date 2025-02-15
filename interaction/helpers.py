from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import base64
import io
import os
from PIL import Image
import subprocess
import numpy as np
import shutil
import torch


#-----------------------------------------------------app preparation------------------------------

# Variable to ensure models are copied only once
models_copied = False  

repo_dir = os.path.abspath("Orginal_CycleGAN_Repository")
interaction_dir = os.path.abspath("interaction")

# Paths for application images
input_path_app = os.path.join(interaction_dir, "images", "input")
upload_path_app = os.path.join(interaction_dir, "images", "upload")
input_image_path_app = os.path.join(input_path_app, "input.png")
output_image_path_app = os.path.join(interaction_dir, "images", "all_images")
upload_image_path_app = os.path.join(upload_path_app, "upload.png")

# Paths for repository images
input_image_path_repo = os.path.join(repo_dir, "datasets", "SketchPad", "testA", "input.png")
model_target = os.path.join(repo_dir, "checkpoints", "SketchPad")
output_image_path_repo = os.path.join(repo_dir, "results/SketchPad/test_latest/images")

# Ensure necessary directories exist
os.makedirs(input_path_app, exist_ok=True)
os.makedirs(output_image_path_app, exist_ok=True)
os.makedirs(upload_path_app, exist_ok=True)
os.makedirs(model_target, exist_ok=True)



def copy_all_models_once():
    """Copies all models from 'interaction/models/' to 'checkpoints/SketchPad/' once."""
    global models_copied

    if models_copied:
        return  # Models already copied

    model_source = os.path.join(interaction_dir, "models") 

    if not os.path.exists(model_source):
        print(f"ERROR: Model directory {model_source} does not exist.")
        return False


    # Copy all models
    for file in os.listdir(model_source):
        shutil.copy(os.path.join(model_source, file), model_target)

    models_copied = True   # Prevent multiple copies
    return True


def get_available_models():
    """Returns a list of available models based on filenames."""
    model_dir = os.path.abspath("Orginal_CycleGAN_Repository/checkpoints/SketchPad")

    if not os.path.exists(model_dir):
        return ["No models found"]

    available_models = [
        file.replace("_net_G.pth", "") for file in os.listdir(model_dir) if file.endswith("_net_G.pth")
    ]

    return available_models if available_models else ["No models found"]



#-----------------------------------------------------image preparation------------------------------
def get_next_image_number(output_image_path_app):
    """Finds the next available number for saving output images."""
    existing_numbers = [int(f.split("_")[0]) for f in os.listdir(output_image_path_app) 
                        if f.endswith("_real.png") and f.split("_")[0].isdigit()]
    return max(existing_numbers, default=0) + 1


# image transformation
def ensure_white_background(image):
    """Ensures that the image has a white background if it has transparency."""
    if image.mode == 'RGBA':
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        background.paste(image, (0, 0), image)
        return background.convert('RGB')
    return image.convert('RGB')
    

def find_bounds(image):
    """Ensures that the image has a white background if transparency exists."""
    img_array = np.asarray(image)

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Identify non-white pixels (white is assumed to be (255, 255, 255))
    non_white_pixels = np.where(np.all(img_array[:, :, :3] != [255, 255, 255], axis=-1))

    if len(non_white_pixels[0]) == 0 or len(non_white_pixels[1]) == 0:
        return None, None, None, None, None  # No content found

    # Determine bounding box coordinates
    top = non_white_pixels[0].min()
    bottom = non_white_pixels[0].max()
    left = non_white_pixels[1].min()
    right = non_white_pixels[1].max()

    # Resize to fit within 245x245 while maintaining aspect ratio
    cropped_img = image.crop((left, top, right + 1, bottom + 1))
    return cropped_img


def process_image(image):
    """Processes the image by cropping content, resizing, and placing it on a white 250x250 background."""
    cropped_img = find_bounds(image)

    if cropped_img is None:
        raise ValueError("No content found in the image.")

    # Resize to fit within 245x245 while maintaining aspect ratio
    cropped_img.thumbnail((245, 245), Image.Resampling.LANCZOS)
    # Create a 250x250 white background and center the image
    background = Image.new('RGB', (256, 256), (255, 255, 255))
    offset = ((256 - cropped_img.size[0]) // 2, (256 - cropped_img.size[1]) // 2)
    background.paste(cropped_img, offset)
    return background


def convert_image_to_black_white_dynamic(image):
    """Converts an image to black and white using a dynamically determined threshold."""
    image = image.convert("L")  # Convert to grayscale
    image_array = np.array(image)

    threshold = 240
    # Apply thresholding
    binary_image_array = np.where(image_array < threshold, 0, 255).astype(np.uint8)

    # Convert back to PIL image
    return Image.fromarray(binary_image_array)

def process_and_save_image(image):
    """Processes an image by cropping, resizing, and centering on a white background."""
    processed_img = process_image(image)
    processed_img = convert_image_to_black_white_dynamic(processed_img)
    processed_img.save(input_image_path_app)
    


#-----------------------------------------------------calling the model for transformation-----------------------------

def rename_model(model_name):
    """Returns a list of available models based on filenames."""
    model_dir = os.path.abspath("Orginal_CycleGAN_Repository/checkpoints/SketchPad")
    
    latest_model = os.path.join(model_dir, "latest_net_G.pth")
    new_model = os.path.join(model_dir, f"{model_name}_net_G.pth")

    # Falls es bereits ein latest_net_G.pth gibt, benenne es zurück
    if os.path.exists(latest_model):
        os.rename(latest_model,new_model)

    # Jetzt das neue Modell als latest_net_G.pth setzen
    else:
        try:
            os.rename(new_model,latest_model)
            print(f"{model_name}_net_G.pth wurde zu latest_net_G.pth umbenannt.")
        except:
            print(f"FEHLER: Datei {model_name}_net_G.pth nicht gefunden.")


def run_model(model_name):
    """Runs the selected model and saves the generated images."""
    os.makedirs(os.path.dirname(input_image_path_repo), exist_ok=True)
    shutil.copy(input_image_path_app, input_image_path_repo)
    gpu_ids = "0" if torch.cuda.is_available() else "-1"  # GPU or CPU


    rename_model(model_name)
    
    # call test.py in the repository
    command = [
    "python", "test.py",
    "--dataroot", "datasets/SketchPad",
    "--name", "SketchPad",
    "--model", "test",
    "--no_dropout",
    "--gpu_ids", gpu_ids,
    ]

    print("Starting model with command:", " ".join(command))
    subprocess.run(command, cwd=repo_dir)

    rename_model(model_name)

    next_image_number = get_next_image_number(output_image_path_app)
    for file in os.listdir(output_image_path_repo):
        if file.endswith("_real.png"):
            new_name = f"{next_image_number}_real.png"
        elif file.endswith("_fake.png"):
            new_name = f"{next_image_number}_fake.png"
            shutil.copy(os.path.join(output_image_path_repo, file), upload_image_path_app)
            print(f"Fake-Bild gespeichert als Upload: {upload_image_path_app}")
        else:
            continue
        
        shutil.copy(os.path.join(output_image_path_repo, file), os.path.join(output_image_path_app, new_name))

    # remove all files to prepare folders for next transformation
    shutil.rmtree(os.path.join(repo_dir, "results", model_name), ignore_errors=True)
    shutil.rmtree(os.path.dirname(input_image_path_repo), ignore_errors=True)
