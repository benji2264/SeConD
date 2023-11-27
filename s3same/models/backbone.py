from functools import partial

import torch
import torch.nn as nn

backbones = {
    # model_name: [torch.hub source, torch.hub name, embedding size]
    "vits16": ["facebookresearch/dino:main", "dino_vits16", 384],
    "resnet18": ["pytorch/vision", "resnet18", 512],
    "resnet50": ["pytorch/vision", "resnet50", 2048],
}

# Check model name
valid_names = list(backbones.keys())


def load_backbone(model_name: str):
    """
    Returns randomly initialized backbone given model name,
    as well as the network's embedding size.
    Args:
        model_name: str, name of the model to load.
            Should be one of the keys of the backbones dict.
    """
    assert (
        model_name in valid_names
    ), f"`model_name` should be one of {valid_names}, got {model_name}"

    # Load from hub
    source, name, embed_size = backbones[model_name]
    load_model = partial(torch.hub.load, source, name)

    if model_name.startswith("resnet"):
        model = load_model(weights=None)
        backbone = nn.Sequential(*list(model.children())[:-1])

    if model_name.startswith("vit"):
        model = load_model(pretrained=False)
        backbone = model

    return backbone, embed_size
