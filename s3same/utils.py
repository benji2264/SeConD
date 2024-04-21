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
            v2.RandomResizedCrop(input_size, antialias=True),
            v2.RandomHorizontalFlip(flip_prob),
            v2.RandomRotation(rot_range),
            v2.ColorJitter(brightness, contrast, saturation, hue),
            normalize_transform,
        ]
    )

    val_transforms = v2.Compose(
        [
            v2.Resize(input_size, antialias=True),
            v2.CenterCrop(input_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize_transform,
        ]
    )
    return train_transforms, val_transforms


def scale_bbox(bbox, scale=0.4, img_size=(720, 1280), xywh_format=True):
    """
    Increases or decreases bbox height and width by a factor of `scale`.
    If box becomes bigger than `img_size`, it's clipped to the img size.
    Args:
        bbox: List, bbox coordinates in format xywh if `xywh_format` is True,
            otherwise format will be xyxy
        scale: float, scaling factor. If scale == 0.2,
            box height and width will be increased by 20%.
            If scale < 0, box will be reduced.
        img_size: Tuple[int], height and width of the image.
    """
    img_h, img_w = img_size
    scale = 1 + scale

    # Retrieve box coordinates
    if xywh_format:
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h

    else:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1

    # Compute new (h, w) and delta
    new_w, new_h = map(lambda x: int(x * scale), [w, h])
    dw, dh = int((new_w - w) / 2), int((new_h - h) / 2)

    # Update bbox and return
    x1, y1 = max(0, x1 - dw), max(0, y1 - dh)
    x2, y2 = min(img_w, x2 + dw), min(img_h, y2 + dh)

    if xywh_format:
        return x1, y1, (x2 - x1), (y2 - y1)
    else:
        return x1, y1, x2, y2
