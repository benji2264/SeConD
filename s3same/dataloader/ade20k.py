import json
import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

AVOID_LABELS = set(
    [
        "building",
        "floor",
        "sky",
        "road",
        "wall",
        "ground",
        "field",
        "grass",
        "ceiling",
        "sidewalk",
        # TODO: continue this list
    ]
)

CROP_SIZE = [128, 256]


class ADE20K(Dataset):

    def __init__(self, path, type="train") -> None:
        if type == "train":
            type = "training"
        elif type == "val":
            type = "validation"
        else:
            raise Exception(f"Type {type} is not supported, choose 'train' or 'val'")
        self.img_dir = os.path.join(path, "images", "ADE", type)
        self.img_infos = self.load_infos()

    def load_infos(self):
        infos = {}
        idx = 0

        # Get the list of the subdirectories
        dirs = [d for d in os.listdir(self.img_dir) if os.path.isdir(os.path.join(self.img_dir, d))]
        for dir in dirs:
            subdir = os.path.join(self.img_dir, dir)
            sub_dirs = [d for d in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, d))]

            for subsubdir in sub_dirs:
                # Get all the jsons in the current subdirectory
                names = [
                    n for n in os.listdir(os.path.join(subdir, subsubdir)) if n.endswith(".json")
                ]

                for name in names:
                    with open(
                        os.path.join(subdir, subsubdir, name), "r", encoding="latin-1"
                    ) as file:
                        data = json.load(file)["annotation"]

                    max_area = -1
                    label = None
                    for region in data["object"]:
                        # Go to the next candidate if the current is from
                        # one of the categories we want to avoid
                        if region["raw_name"] in AVOID_LABELS:
                            continue

                        plgnX = np.array(region["polygon"]["x"])
                        minX = np.min(plgnX)
                        maxX = np.max(plgnX)
                        plgnY = np.array(region["polygon"]["y"])
                        minY = np.min(plgnY)
                        maxY = np.max(plgnY)

                        area = (maxX - minX) * (maxY - minY)

                        # Update the 'best' crop if the new one is bigger
                        if area > max_area:
                            max_area = area
                            label = region["raw_name"]
                            corner1 = (minX, minY)
                            corner2 = (maxX, maxY)

                    # In case there is no crop in the categories of interest,
                    # go to the next image
                    if label is None:
                        continue

                    # Store the informations about the 'best' crop
                    infos[idx] = {
                        "label": label,
                        "corner1": corner1,
                        "corner2": corner2,
                        "path": os.path.join(subdir, subsubdir, data["filename"]),
                    }
                    idx += 1
        return infos

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        # Get the infos about the crop
        img_path = self.img_infos[idx]["path"]
        minX, minY = self.img_infos[idx]["corner1"]
        maxX, maxY = self.img_infos[idx]["corner2"]
        label = self.img_infos[idx]["label"]

        # Load the image, crop & resize it
        image = read_image(img_path)
        image = v2.functional.resized_crop(
            image, minY, minX, maxY - minY, maxX - minX, size=CROP_SIZE
        )

        return image, label


if __name__ == "__main__":
    path = "../../Datasets/ADE20K_2021_17_01"
    ade20k_train = ADE20K(path, "train")
    ade20k_val = ADE20K(path, "val")

    train_dataloader = DataLoader(ade20k_train, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(ade20k_val, batch_size=64, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {len(train_labels)}")
    print(train_labels)
    n = np.random.randint(64)
    img = np.transpose(train_features[n].squeeze(), axes=(1, 2, 0))
    label = train_labels[n]
    print(label)
    plt.imshow(img)
    plt.show()
