"""
Script to convert weights for a ResNet-18 LightningModule with a torchvision backbone 
to our new naming convention, as defined by the `models/build_model.py` file.
"""

import argparse
from tqdm import tqdm
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        "-i",
        required=True,
        type=str,
        help="Path to torchvision .pt checkpoint file.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        required=True,
        type=str,
        help="Path where to save the converted weights.",
    )
    args = parser.parse_args()

    state_dict = torch.load(args.input_path, map_location=torch.device("cpu"))[
        "state_dict"
    ]
    spec_names = {
        # stem layers
        # "model": "backbone",
        "model.conv1.weight": "backbone.0.weight",
        "model.bn1.weight": "backbone.1.weight",
        "model.bn1.bias": "backbone.1.bias",
        "model.bn1.running_mean": "backbone.1.running_mean",
        "model.bn1.running_var": "backbone.1.running_var",
        # final layers
        "model.fc.weight": "linear.weight",
        "model.fc.bias": "linear.bias",
    }

    new_state_dict = {}
    prefix = "model.layer"

    for name, mod in tqdm(state_dict.items()):
        if name == "model.fc.weight":
            print("mod.shape", mod.shape)
        old_name = name
        if name in spec_names:
            new_name = spec_names[name]

        elif name.startswith(prefix):
            name = name.replace(prefix, "")
            layer_idx = int(name[0]) + 3
            new_name = f"backbone.{layer_idx}" + name[1:]
        else:
            continue
        new_state_dict[new_name] = mod

    torch.save(new_state_dict, args.output_path)
    print(f"Done! New checkpoint saved to {args.output_path}")


import torch
import torchvision.models as models

# Step 1: Initialize the model (if starting with a pretrained model)
# model = models.resnet18(pretrained=True)

checkpoint_path = "jobs/GPU_backbone_training_A40_epch40/tb_logs/coco_r18_teacher_cs_v1/version_0/checkpoints/epoch=39-step=9200.ckpt"

# Step 2: Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Extract the state dictionary
state_dict = checkpoint["state_dict"]

# Create a new state dictionary with modifications
new_state_dict = {}
for key in state_dict.keys():
    new_key = key
    # Rename keys by adding a prefix or changing the key structure
    if key.startswith("model."):
        new_key = key[len("model.") :]

    if key.startswith("model.fc."):
        continue
    # Add the modified key to the new dictionary
    new_state_dict[new_key] = state_dict[key]

# Initialize a fresh model
model = models.resnet18()

# Load the modified state dictionary into the model
model.load_state_dict(new_state_dict, strict=False)

# Prepare the new checkpoint with the modified state dictionary
new_checkpoint = {
    "state_dict": model.state_dict(),
    "epoch": checkpoint.get("epoch", None),  # Keep the original epoch if it exists
    "optimizer": checkpoint.get(
        "optimizer", None
    ),  # Keep the original optimizer state if it exists
}

# Save the new checkpoint
torch.save(new_checkpoint, "models/r18_coco_renamed.pth")
