import os
import gc
import argparse
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from utils import *

random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

sam_model = "vit_h"
output_path = "./eaxle_gs_masks_csv_with_id"


def get_masks_hierarchy(masks, iou_threshold=0.9):
    masks_sorted = sorted(masks, key=lambda x: x["area"], reverse=True)
    masks_results = []
    obj_id = 0

    while len(masks_sorted) > 0:
        # Select one region (biggest one first)
        mask = masks_sorted[0]

        # Go through all other regions
        for i, other_mask in enumerate(masks_sorted):
            # if other_mask included in mask, assign them to the same object
            m1, m2 = mask["segmentation"], other_mask["segmentation"]
            inter = np.logical_and(m1, m2).sum()
            if inter > m2.sum() * iou_threshold:
                other_mask["obj_id"] = obj_id
                masks_results.append(other_mask)

        # Pop all masks with the same object id
        masks_sorted = [m for m in masks_sorted if "obj_id" not in m]

        # Go to next object
        obj_id += 1

    return masks_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", required=True, type=str, help="Path to dataset training images."
    )
    parser.add_argument(
        "--output",
        required=False,
        type=str,
        help="Where to save the CSV files with the masks.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        type=str,
        help="Path to HQ-SAM ViT-H checkpoint.",
    )
    args = parser.parse_args()

    output_path = args.output

    if not args.output:
        output_path = "masks_outputs"
    os.makedirs(output_path, exist_ok=True)

    nb_images = None  # max nb of images, None for all of them
    # nb_images = 30

    # Get image paths in folder
    filepaths = find_files(args.data, exts=["jpg", "png", "JPEG"])

    # Shuffle and get `nb_images` of them
    random.shuffle(filepaths)
    if nb_images:
        filepaths = filepaths[:nb_images]

    # Segment
    sam_h = SAM(sam_model, args.ckpt, highdetails=False)

    # for img_path, img in tqdm(list(zip(filepaths, all_imgs))):
    for img_path in tqdm(filepaths):
        img = load_image(img_path)

        # Skip image if results already exist
        result_filename = img_path.split("/")[-1].split(".")[0] + ".csv"
        output_file = os.path.join(output_path, result_filename)
        if os.path.exists(output_file):
            continue

        data = {}
        data["img_path"] = []
        data["h"] = []
        data["w"] = []
        masks = sam_h(img)
        masks = get_masks_hierarchy(masks)

        for mask in masks:
            for key, val in mask.items():
                if key in data:
                    data[key].append(val)
                elif key != "segmentation":
                    data[key] = [val]

                if key == "segmentation":
                    data["h"].append(val.shape[0])
                    data["w"].append(val.shape[1])
            data["img_path"].append(img_path)

        data_df = pd.DataFrame(data)
        data_df["area_pct"] = data_df["area"] / (data_df["h"] * data_df["w"])

        data_df.to_csv(output_file, index=False)
        gc.collect()
