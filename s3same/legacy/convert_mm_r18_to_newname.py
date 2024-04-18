"""
Script to convert weights for a ResNet-18 LightningModule with a torchvision backbone 
to our new naming convention, as defined by the `models/build_model.py` file.
"""

import argparse
import torch

NUM_CLASSES = 200  # 200 for COCO, 35 for Cityscapes etc...

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
    print(torch.load(args.input_path, map_location=torch.device("cpu")).keys())
    from models.build_model import SupervisedClassifier

    spec_names = {
        # stem layers
        # "model": "backbone",
        "conv1.weight": "backbone.0.weight",
        "bn1.weight": "backbone.1.weight",
        "bn1.bias": "backbone.1.bias",
        "bn1.running_mean": "backbone.1.running_mean",
        "bn1.running_var": "backbone.1.running_var",
        # final layers
        "fc.weight": "linear.weight",
        "fc.bias": "linear.bias",
    }

    # for name, mod in state_dict.items():
    #     print(f"{name:60} {mod.shape}")

    # model = ResNet18Module.load_from_checkpoint(
    #     args.input_path, num_classes=NUM_CLASSES
    # )

    # new_state_dict = []
    # for name, mod in model.named_modules:
    #     new_name = "model." + name
    #     new_state_dict[new_name] = mod
    #     print(f"{name} -> new_name")
    # torch.save(new_state_dict, args.output_path)

    new_state_dict = {}
    prefix = "layer"

    for name, mod in state_dict.items():
        print(name)
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
        print(f"{old_name} -> {new_name}")
    torch.save(new_state_dict, args.output_path)

    model = SupervisedClassifier("resnet18", num_classes=200)
    model.load_state_dict(torch.load(args.output_path))
