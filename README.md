# CycleGAN: Sketch-to-Real Image Translation

## Overview
This repository provides the necessary setup and tools to utilize a CycleGAN model for converting sketches into realistic images. The actual CycleGAN implementation is included as a submodule, while this repository focuses on its application, usage, and insights gained from working with it.

CycleGAN is a type of Generative Adversarial Network (GAN) that enables image-to-image translation without requiring paired training data. This makes it useful for transforming hand-drawn sketches into realistic images, even when exact input-output pairs are not available.

## Features
- Provides an easy-to-use setup for applying CycleGAN to sketch-to-real image translation.
- Includes the CycleGAN model as a submodule.
- Offers scripts and configurations for running and evaluating the model.
- Allows the use of custom datasets.
- Supports GPU acceleration for faster processing.

## Setup and Usage

### Clone this repository and initialize the submodule
First, clone this repository and initialize the CycleGAN submodule, which contains the actual model implementation.

```bash
git clone https://github.com/Isabelle-Gbl/Sketch2RealImage-with-CycleGAN.git
cd Sketch2RealImage-with-CycleGAN
git submodule update --init --recursive
