from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import base64
import io
import os
from PIL import Image
import subprocess
import numpy as np
import shutil


repo_dir = os.path.abspath("Orginal_CycleGAN_Repository")
interaction_dir = os.path.abspath("interaction")

# 🔹 Variablen für die Pfade in der App
input_image_path_app = os.path.join(interaction_dir, "images", "input", "input.png")
output_image_path_app = os.path.join(interaction_dir, "images", "all_images")
upload_image_path_app = os.path.join(interaction_dir, "images", "upload", "upload.png")

# 🔹 Variablen für die Pfade im Repository
input_image_path_repo = os.path.join(repo_dir, "datasets", "SketchPad", "testA", "input.png")
model_target = os.path.join(repo_dir, "checkpoints", "SketchPad")
output_image_path_repo = os.path.join(repo_dir, "results/SketchPad/test_latest/images")


os.makedirs(model_target, exist_ok=True)

def copy_selected_model(model_name):
    """Kopiert das ausgewählte Modell aus interaction/models/ nach SketchPad/."""
    repo_dir = os.path.abspath("Orginal_CycleGAN_Repository")
    interaction_dir = os.path.abspath("interaction")
    
    model_target = os.path.join(repo_dir, "checkpoints", "SketchPad")
    model_source = os.path.join(interaction_dir, "models", model_name)  # Ausgewähltes Modell
    
    if not os.path.exists(model_source):
        print(f"FEHLER: Das Modell '{model_name}' existiert nicht in {model_source}")
        return False
    
    # Ordner vorher leeren
    shutil.rmtree(model_target)
    os.makedirs(model_target, exist_ok=True)

    # Dateien aus dem gewählten Modell kopieren (ohne den übergeordneten Ordner)
    for file in os.listdir(model_source):
        shutil.copy(os.path.join(model_source, file), model_target)

    print(f"Modell '{model_name}' kopiert nach {model_target}")
    return True

def get_available_models():
    """
    Ruft die Namen aller verfügbaren Modelle aus dem SketchPad-Verzeichnis ab.
    Ein Modell wird erkannt, wenn in seinem Ordner eine Datei "latest_net_G.pth" existiert.
    """
    model_dir = "interaction/models"
    
    if not os.path.exists(model_dir):
        return ["No models found"]
    
    available_models = []
    for folder in os.listdir(model_dir):
        folder_path = os.path.join(model_dir, folder)
        model_path = os.path.join(folder_path, "latest_net_G.pth")

        if os.path.isdir(folder_path) and os.path.exists(model_path):
            available_models.append(folder)

    return available_models if available_models else ["No models found"]


# image transformation
def ensure_white_background(image):
    """Ensures that the image has a white background if it has transparency."""
    if image.mode == 'RGBA':
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        background.paste(image, (0, 0), image)
        return background.convert('RGB')
    return image.convert('RGB')
    

def find_bounds(image):
    """
    Finds the bounding box of non-white pixels in an image and crops it.
    Returns:
        Cropped version of the image or None if no content is found.
    """
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
    """
    Converts a PIL image to pure black and white based on a dynamically determined threshold.
    The threshold is set as the midpoint between the minimum and maximum pixel values.
    
    Parameters:
        image (PIL.Image): Input image in RGB format.
    
    Returns:
        PIL.Image: Binarized black and white image.
    """
    image = image.convert("L")  # Convert to grayscale
    image_array = np.array(image)

    threshold = 240
    # Apply thresholding
    binary_image_array = np.where(image_array < threshold, 0, 255).astype(np.uint8)

    # Convert back to PIL image
    return Image.fromarray(binary_image_array)





def process_and_save_image(image):
    """Processes and saves an image for input/output."""
    processed_img = process_image(image)
    processed_img = convert_image_to_black_white_dynamic(processed_img)
    processed_img.save(input_image_path_app)
    



def get_next_image_number(output_image_path_app):
    """Findet die nächste verfügbare Nummer für das Speichern der Ausgabe-Bilder."""
    existing_numbers = [int(f.split("_")[0]) for f in os.listdir(output_image_path_app) 
                        if f.endswith("_real.png") and f.split("_")[0].isdigit()]
    return max(existing_numbers, default=0) + 1

def run_model(model_name):
    """Hauptfunktion, die das Modell ausführt und die Bilder speichert."""
    # 1️⃣ Sicherstellen, dass die Zielordner existieren
    copy_selected_model(model_name)
    os.makedirs(os.path.dirname(input_image_path_repo), exist_ok=True)
    
    # 2️⃣ Input-Bild ins Repository kopieren
    shutil.copy(input_image_path_app, input_image_path_repo)
    # print(f"Input-Bild kopiert: {input_image_path_app} -> {input_image_path_repo}")

    # 3️⃣ Modell starten
    command = [
        "python", "test.py",
        "--dataroot", "datasets/SketchPad",
        "--name", "SketchPad",
        "--model", "test",
        "--no_dropout"
    ]
    print("Starte Modell mit Befehl:", " ".join(command))
    subprocess.run(command, cwd=repo_dir)

    # # 4️⃣ Ergebnisse aus dem Modell holen
    # if not os.path.exists(output_image_path_repo):
    #     print(f"FEHLER: Modell-Ergebnisordner existiert nicht: {output_image_path_repo}")
    #     return
    
    next_image_number = get_next_image_number(output_image_path_app)

    for file in os.listdir(output_image_path_repo):
        if file.endswith("_real.png"):
            new_name = f"{next_image_number}_real.png"
        elif file.endswith("_fake.png"):
            new_name = f"{next_image_number}_fake.png"
            shutil.copy(os.path.join(output_image_path_repo, file), upload_image_path_app)  # Upload-Bild speichern
            print(f"Fake-Bild gespeichert als Upload: {upload_image_path_app}")
        else:
            continue
        
        shutil.copy(os.path.join(output_image_path_repo, file), os.path.join(output_image_path_app, new_name))
        print(f"Ergebnis gespeichert: {file} -> {new_name}")

    # 5️⃣ Aufräumen: Ergebnisse & Input aus dem Repository löschen
    shutil.rmtree(os.path.join(repo_dir, "results", model_name), ignore_errors=True)
    shutil.rmtree(os.path.dirname(input_image_path_repo), ignore_errors=True)
    # print(f"Aufgeräumt: {output_image_path_repo} & {input_image_path_repo}")

