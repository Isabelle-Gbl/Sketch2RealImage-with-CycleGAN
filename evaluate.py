"""
CycleGAN Model Evaluation Script

This script automates the evaluation of multiple trained CycleGAN models on a specified test dataset.
It systematically loads different model checkpoints, runs inference using the `test.py` script, and organizes the results.

## Functionality:
1. Locates all trained model checkpoints (`*_net_G_A.pth` or `*_net_G_B.pth`) in the specified model folder.
2. Iterates through each model, replacing `latest_net_G.pth` with the current checkpoint.
3. Runs `test.py` within the original CycleGAN repository using the appropriate GPU or CPU.
4. Saves the results in a structured folder format.
5. Copies important model-related files (`train_opt.txt`, `test_opt.txt`, `loss_log.txt`).
6. Moves the tested checkpoint into a `model` subfolder for documentation.

## Usage:
Run the script from the command line:

    python evaluate_models.py <motive_name> <testset> <model_folder> <generator>
    

### Parameters:
- `<motive_name>`: Name of the dataset/motive (e.g., `"car"`, `"butterfly"`).
- `<testset>`: Test set to use (`"A"` or `"B"`).
- `<model_folder>`: Folder containing the trained models (e.g., `"car_cyclegan"`).
- `<generator>`: Which generator to evaluate (`"G_A"` or `"G_B"`).

### Example:
    python evaluate_models.py car A car_cyclegan G_A

This will:
- Find all `*_net_G_A.pth` models in `checkpoints/car_cyclegan`
- Run inference on `datasets/car/testA`
- Save results in `results/car_cyclegan/{model_checkpoint_name}`
- Move tested checkpoints to `results/car_cyclegan/test_latest/model`
"""

import os
import shutil
import subprocess
import argparse
import torch

def evaluate_models(motive_name, testset, model_folder, generator):

    repo_dir = os.path.abspath("Orginal_CycleGAN_Repository")
    dataroot_path = f"datasets/{motive_name}/test{testset}"
    results_path = f"{repo_dir}/results/{model_folder}/test_latest"
    repo_model_folder_path = os.path.join(repo_dir,f"checkpoints/{model_folder}")

    # Find all relevant model files
    models = [f for f in os.listdir(repo_model_folder_path) if f.endswith(f"_net_{generator}.pth")]

    if not models:
        print("No matching models found.")
        return


    for mod_i in models:
        mod_i_path = os.path.join(repo_model_folder_path, mod_i)
        mod_i_copy_path = os.path.join(repo_model_folder_path, "latest_net_G.pth")

        # Create a backup of the current model
        shutil.copy(mod_i_path, mod_i_copy_path)

        # Determine whether to use GPU or CPU
        gpu_ids = "0" if torch.cuda.is_available() else "-1"

        # Run inference using the current model checkpoint
        command = [
            "python", "test.py",
            "--dataroot", dataroot_path,
            "--name", model_folder,
            "--model", "test",
            "--no_dropout",
            "--gpu_ids", gpu_ids,
        ]
        print(command)
        subprocess.run(command, cwd=repo_dir, check=True)

        # Ensure that the test folder actually exists
        if not os.path.exists(results_path):
            print(f"Warning: {results_path} does not exist, could not be renamed.")
            continue  # Proceed to the next iteration

        # Create a new unique name for the folder
        mod_i_name = mod_i.replace(".pth", "")
        mod_result_name = f"{mod_i_name}_{model_folder}" if len(model_folder) <= 20 else mod_i_name
        mod_result_path = f"{repo_dir}/results/{model_folder}/{mod_result_name}"
        
        counter = 2
        while os.path.exists(mod_result_path):
            mod_result_path = f"{repo_dir}/results/{model_folder}/{mod_result_name}_{counter}"
            counter += 1

        # Rename `test_latest` folder to the specific model folder
        os.rename(results_path, mod_result_path)
        print(f"{results_path} renamed to {mod_result_path}")

    print("Transformation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic model run")
    parser.add_argument("motive_name", type=str, help="Name of the motive")
    parser.add_argument("testset", type=str, choices=["A", "B"], help="Test set (A or B)")
    parser.add_argument("model_folder", type=str, help="Name of the model folder")
    parser.add_argument("generator", type=str, choices=["G_A", "G_B"], help="Generator (G_A or G_B)")

    args = parser.parse_args()
    evaluate_models(args.motive_name, args.testset, args.model_folder, args.generator)
