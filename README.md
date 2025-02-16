# CycleGAN: Sketch-to-Real Image Translation

<picture>
  <img src="https://github.com/user-attachments/assets/0d2052e8-4028-4584-94e1-2c5b4dc06d5d" width="800">
</picture>

## Overview
This repository provides the necessary setup and tools to utilize a CycleGAN model for converting sketches into realistic images. The actual CycleGAN implementation is included as a submodule, while this repository focuses on its application, usage, and insights gained from working with it.

<picture>
  <img src="https://github.com/user-attachments/assets/c81968d6-74ef-46bc-98b7-4bcfd6e62173" width="800">
</picture>

CycleGAN is a type of Generative Adversarial Network (GAN) that enables image-to-image translation without requiring paired training data. This makes it useful for transforming hand-drawn sketches into realistic images, even when exact input-output pairs are not available.

## Features
- Provides an easy-to-use setup for applying CycleGAN to sketch-to-real image translation.
- Includes the CycleGAN model as a submodule.
- Offers scripts and configurations for running and evaluating the model.

## Setup and Usage

### Clone this repository and initialize the submodule
First, clone this repository and initialize the CycleGAN submodule, which contains the actual model implementation.

```bash
git clone https://github.com/Isabelle-Gbl/Sketch2RealImage-with-CycleGAN.git
cd Sketch2RealImage-with-CycleGAN
git submodule update --init --recursive
```

## Adding a Sketch Dataset
To train the model, you need a dataset of sketches. A good option is the [Google QuickDraw dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified), which contains thousands of categorized sketches. This is only for the class "A", you will have to prepare a dataset for the real images class "B" by yourself.

### Steps to add the dataset:
1. **Download the `.ndjson` files** from the QuickDraw dataset (Google account required).
2. **Move the downloaded file** into a dataset folder with your chosen class name.
3. **Convert the `.ndjson` file into images** using the provided script:
```bash
python prepare_ndjson.py <class_name> <image_count> --root_path <root_path>
```
Example:
```bash
python prepare_ndjson.py car 4000
```
This extracts 4000 images from the dataset and places them in the appropriate folder for training.

## Start Training
To monitor the training process, start a Visdom server in the background:
```bash
python -m visdom.server
```
Then open `http://localhost:8097` in a browser to visualize training progress.

### Running the training script
Use the following command to train the model on your dataset:
```bash
python train.py --dataroot ./datasets/<class_name> --name <your_project_name> --model cycle_gan
```
Example:
```bash
python train.py --dataroot ./datasets/car --name car_cyclegan --model cycle_gan
```

If your system does not have CUDA support, run:
```bash
python train.py --dataroot ./datasets/<class_name> --name car_cyclegan --model cycle_gan --gpu_ids -1
```
This will use the CPU instead, though training will be significantly slower.


### Evaluating Models
After training, you may want to compare different versions of the model to see which performs best. You can evaluate a model using the following command:
```bash
python evaluate.py <class_name> <A or B> car_cyclegan <G_A or G_B>
```

Example:
```bash
python evaluate.py car A car_cyclegan G_A
```
This example would use all G_A generators in the checkpoint folder car_cyclegan to transform each image of the folder testA in the car folder. You would find the result in the submodul folder in results/car_cyclegan.

## Web Application (Flask Sketch Pad)
This repository includes a Flask-based web application that allows you to interact with trained models by drawing sketches and converting them into realistic images.

![Sketchpad](https://github.com/user-attachments/assets/279edd77-d87d-4cef-9055-e8d2ae6293fa)

### Setup
To use the web app:
1. Move trained models to `interaction/models`. You have to create the folder called "models". Make sure the model filenames end with `_net_G.pth`.
2. Start the application:
```bash
python app.py
```
3. Open `http://localhost:5000/` in a browser.
4. Select a model from the dropdown menu, draw on the canvas, and click "Transform" to generate images

### Notes on Model Selection
- The models in interaction/models should be named clearly (e.g., car_net_G.pth, butterfly_net_G.pth).
- The Flask app automatically detects models placed in this folder.
- Generated images are stored in interaction/images/all_images, so you can revisit previous results.
