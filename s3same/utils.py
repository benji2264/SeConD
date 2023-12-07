from typing import Union

import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch

from segment_anything_hq import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
)


def find_files(folder, exts=["jpg", "png", "JPEG"]):
    """
    Returns all filenames in folder (recursively)
    ending with one of the given extensions.
    Returns a list of paths starting from the root.
    """
    exts = tuple(exts)
    all_filenames = []
    for root, _, files in os.walk(folder):
        all_filenames += [
            os.path.join(root, basename)
            for basename in files
            if basename.endswith(exts)
        ]
    return all_filenames


def load_image(img: Union[str, np.ndarray]):
    """
    Loads, preprocesses and returns image.
    Args:
        img: str or np array of shape (h, w, 3)
            if str: path to image else image array
    """
    if isinstance(img, str):
        img = np.array(Image.open(img))

    # Remove alpha channel if any
    if len(img.shape) > 2 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    return img


class SAM:
    def __init__(self, vit_model: str, checkpoint: str, highdetails: bool = False):
        """
        Wrapper around SAM  or HQ-SAM for easier use.
        Args:
            vit_model: str, name of ViT backbone to use
                e.g. "vit_h" or "vit_tiny".
            checkpoint: str, path to SAM checkpoint.
            highdetails: bool, if True, SAM will be run on
                crop of the image to segment smaller objects (e.g. screws)
                (hurts inference significantly)
        """
        print("Loading SAM...")
        self.model = sam_model_registry[vit_model](checkpoint=checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        if not highdetails:
            mask_generator = SamAutomaticMaskGenerator(self.model)
        else:
            mask_generator = SamAutomaticMaskGenerator(
                model=self.modelmodel,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=2,
                crop_n_points_downscale_factor=2,
                #     min_mask_region_area=100,  # Requires open-cv to run post-processing
            )
        self.mask_generator = mask_generator
        print("Done.")

    def get_region_from_point(self, masks, x, y):
        candidate_masks = []
        for mask in masks:
            if mask["segmentation"][y, x]:
                candidate_masks.append(mask)
        candidate_masks = sorted(candidate_masks, key=(lambda x: x["area"]))
        return candidate_masks[0]

    def show_anns(self, anns, ax=None):
        ax = plt.gca() if not ax else ax
        h, w = anns[0]["segmentation"].shape[:2]
        img = np.ones((h, w, 4))
        img[:, :, 3] = 0
        for mask in anns:
            m = mask["segmentation"]
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)

    def __call__(self, image):
        """
        SAM inference on input image.
        Args:
            image: np array of shape (h,w,3), image to segment.
        """
        masks = self.mask_generator.generate(image)
        masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
        return masks
