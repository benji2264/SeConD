import torch
from torchvision.transforms import v2
from lightly.transforms.utils import IMAGENET_NORMALIZE


def get_device():
    """
    Auto choose between `cuda`, `mps` and `cpu`
    (in this order of preference)
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps():
        device = "mps"
    else:
        device = "cpu"
    return device


def get_transforms(
    input_size,
    flip_prob=0.5,
    rot_range=(-10, 10),
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.1,
):
    """
    Common transforms for teacher pretraining and self-distillation
    """
    normalize_transform = v2.Normalize(
        mean=IMAGENET_NORMALIZE["mean"],
        std=IMAGENET_NORMALIZE["std"],
    )

    train_transforms = v2.Compose(
        [
            v2.RandomResizedCrop(input_size),
            v2.RandomHorizontalFlip(flip_prob),
            v2.RandomRotation(rot_range),
            v2.ColorJitter(brightness, contrast, saturation, hue),
            normalize_transform,
        ]
    )

    val_transforms = v2.Compose(
        [
            v2.Resize(input_size),
            v2.CenterCrop(input_size),
            v2.ToTensor(),
            normalize_transform,
        ]
    )
    return train_transforms, val_transforms
