import json
import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

LABEL_TO_ID = {
  'unlabeled'           : 0,
  'ego vehicle'         : 1,
  'rectification border': 2,
  'out of roi'          : 3,
  'static'              : 4,
  'dynamic'             : 5,
  'ground'              : 6,
  'road'                : 7,
  'sidewalk'            : 8,
  'parking'             : 9,
  'rail track'          : 10,
  'building'            : 11,
  'wall'                : 12,
  'fence'               : 13,
  'guard rail'          : 14,
  'bridge'              : 15,
  'tunnel'              : 16,
  'pole'                : 17,
  'polegroup'           : 18,
  'traffic light'       : 19,
  'traffic sign'        : 20,
  'vegetation'          : 21,
  'terrain'             : 22,
  'sky'                 : 23,
  'person'              : 24,
  'persongroup'         : 24,
  'rider'               : 25,
  'car'                 : 26,
  'cargroup'            : 26,
  'truck'               : 27,
  'bus'                 : 28,
  'caravan'             : 29,
  'trailer'             : 30,
  'train'               : 31,
  'motorcycle'          : 32,
  'motorcyclegroup'     : 32,
  'bicycle'             : 33,
  'bicyclegroup'        : 33,
  'license plate'       : 34,
}

AVOID_LABELS = set(
    [
        "out of roi",
        "road",
        "wall",
        "ground",
        "rectification border",
        "dynamic",
        "sky",
        "static",
        "terrain",
        "vegetation",
        "building",
        "sidewalk",
        "ego vehicle",
    ]
)

CROP_SIZE = [128, 256]


class CityScapes(Dataset):

    def __init__(self, path, type="train", transform=None) -> None:
        self.annot_dir = os.path.join(path, "gtFine_trainvaltest", "gtFine", type)
        # self.annot_dir = "/Users/benjaminmissaoui/Desktop/gt_s24/6476/s3same/s3same/datasets/cityscapes/"
        self.img_dir = os.path.join(path, "leftImg8bit_trainvaltest", "leftImg8bit", type)
        # self.img_dir = "/Users/benjaminmissaoui/Desktop/gt_s24/6476/s3same/s3same/datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train"
        self.img_infos = self.load_infos()
        self.transform = transform

    def load_infos(self):
        infos = {}
        idx = 0

        # Get the list of the subdirectories
        dirs = [
            d for d in os.listdir(self.annot_dir) if os.path.isdir(os.path.join(self.annot_dir, d))
        ]
        for dir in dirs:
            subdir = os.path.join(self.annot_dir, dir)

            # Get all the jsons in the current subdirectory
            names = [n.rpartition("_")[0] for n in os.listdir(subdir) if n.endswith(".json")]

            for name in names:
                with open(os.path.join(subdir, name + "_polygons.json"), "r") as file:
                    objects = json.load(file)["objects"]

                max_area = -1
                label = None
                for region in objects:
                    # Go to the next candidate if the current is from
                    # one of the categories we want to avoid
                    if region["label"] in AVOID_LABELS:
                        continue

                    polygon = np.array(region["polygon"])
                    minX, minY = np.min(polygon, axis=0)
                    maxX, maxY = np.max(polygon, axis=0)

                    area = (maxX - minX) * (maxY - minY)

                    # Update the 'best' crop if the new one is bigger
                    if area > max_area:
                        max_area = area
                        label = region["label"]
                        corner1 = (minX, minY)
                        corner2 = (maxX, maxY)

                # In case there is no crop in the categories of interest,
                # go to the next image
                if label is None:
                    continue

                # Store the informations about the 'best' crop
                infos[idx] = {
                    "label": LABEL_TO_ID[label],
                    "corner1": corner1,
                    "corner2": corner2,
                    "path": os.path.join(
                        self.img_dir, dir, name.rpartition("_")[0] + "_leftImg8bit.png"
                    ),
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
        ) / 255.0
        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    path = "../datasets/cityscapes/"
    # path = "../../Datasets/CityScapes"


    cityscapes_train = CityScapes(path, "train")
    cityscapes_val = CityScapes(path, "val")

    train_dataloader = DataLoader(cityscapes_train, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(cityscapes_val, batch_size=64, shuffle=True)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {len(train_labels)}")
    print(train_labels)
    n = np.random.randint(64)
    img = np.transpose(train_features[n].squeeze(), axes=(1, 2, 0))
    label = train_labels[n]
    print(label)
    print("unique labels = ", np.unique(list(train_labels)))
    print("max(labels) = ", max(train_labels))
    plt.imshow(img)
    plt.show()
