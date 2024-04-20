import json
import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms.v2.functional import resized_crop

from utils import scale_bbox


AVOID_LABELS = set(
    [
        118,  # floor-wood
        125,  # gravel
        144,  # platform
        145,  # playingfield
        148,  # river
        149,  # road
        154,  # sand
        155,  # sea
        159,  # snow
        171,  # wall-brick
        175,  # wall-stone
        176,  # wall-tile
        177,  # wall-wood
        178,  # water-other
        184,  # tree-merged
        186,  # ceiling-merged
        187,  # sky-other-merged
        190,  # floor-other-merged
        191,  # pavement-merged
        192,  # mountain-merged
        193,  # grass-merged
        194,  # dirt-merged
        195,  # paper-merged
        197,  # building-other-merged
        198,  # rock-merged
        199,  # wall-other-merged
        200,  # rug-merged
    ]
)

CROP_SIZE = [128, 256]

ID_TO_LABEL = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
    92: "banner",
    93: "blanket",
    95: "bridge",
    100: "cardboard",
    107: "counter",
    109: "curtain",
    112: "door-stuff",
    118: "floor-wood",
    119: "flower",
    122: "fruit",
    125: "gravel",
    128: "house",
    130: "light",
    133: "mirror-stuff",
    138: "net",
    141: "pillow",
    144: "platform",
    145: "playingfield",
    147: "railroad",
    148: "river",
    149: "road",
    151: "roof",
    154: "sand",
    155: "sea",
    156: "shelf",
    159: "snow",
    161: "stairs",
    166: "tent",
    168: "towel",
    171: "wall-brick",
    175: "wall-stone",
    176: "wall-tile",
    177: "wall-wood",
    178: "water-other",
    180: "window-blind",
    181: "window-other",
    184: "tree-merged",
    185: "fence-merged",
    186: "ceiling-merged",
    187: "sky-other-merged",
    188: "cabinet-merged",
    189: "table-merged",
    190: "floor-other-merged",
    191: "pavement-merged",
    192: "mountain-merged",
    193: "grass-merged",
    194: "dirt-merged",
    195: "paper-merged",
    196: "food-other-merged",
    197: "building-other-merged",
    198: "rock-merged",
    199: "wall-other-merged",
    200: "rug-merged",
}


class COCO(Dataset):

    def __init__(
        self, path, type="train", transform=None, scale_factor=None, nb_views=1
    ) -> None:
        assert nb_views > 0, f"`nb_views` should be at least 1, got {nb_views}"
        self.num_classes = 200
        self.img_dir = os.path.join(path, type + "2017")
        self.json_dir = os.path.join(path, "annotations/panoptic_" + type + "2017.json")
        self.img_infos = self.load_infos()
        self.transform = transform
        self.scale_factor = scale_factor
        self.nb_views = nb_views

    def load_infos(self):
        infos = {}
        idx = 0

        with open(self.json_dir, "r") as file:
            annot = json.load(file)

        for entry in annot["annotations"]:
            max_area = -1
            label = None
            for crop in entry["segments_info"]:
                # Go to the next candidate if the current is from
                # one of the categories we want to avoid
                if crop["category_id"] in AVOID_LABELS:
                    continue
                # Update the 'best' crop if the new one is bigger
                if crop["area"] > max_area:
                    max_area = crop["area"]
                    x, y, width, height = crop["bbox"]
                    label = crop["category_id"]

            # In case there is no crop in the categories of interest,
            # go to the next image
            if label is None:
                continue

            # Store the informations about the 'best' crop
            infos[idx] = {
                "label": label,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "path": os.path.join(
                    self.img_dir, entry["file_name"].split(".")[0] + ".jpg"
                ),
            }
            idx += 1
        return infos

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        # Get the infos about the crop
        img_path = self.img_infos[idx]["path"]
        x = self.img_infos[idx]["x"]
        y = self.img_infos[idx]["y"]
        w = self.img_infos[idx]["width"]
        h = self.img_infos[idx]["height"]
        label = self.img_infos[idx]["label"]

        # Load the image
        image = read_image(img_path)
        img_h, img_w = image.shape[-2:]

        # Handle the case of gray images
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        # Scale bbox
        if self.scale_factor is not None:
            x, y, w, h = scale_bbox(
                [x, y, w, h], scale=self.scale_factor, img_size=(img_h, img_w)
            )

        # Crop to bbox
        image = resized_crop(image, y, x, h, w, size=CROP_SIZE, antialias=True) / 255.0

        # Apply transforms
        tr = self.transform if self.transform is not None else lambda x: x
        images = [tr(image) for _ in range(self.nb_views)]
        labels = [label] * self.nb_views

        if self.nb_views == 1:
            images, labels = images[0], labels[0]

        return images, labels


if __name__ == "__main__":
    path = "../datasets/coco/"
    # path = "../../Datasets/COCO"
    # coco_train = COCO(path, "train")
    coco_val = COCO(path, "val")

    # train_dataloader = DataLoader(coco_train, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(coco_val, batch_size=64, shuffle=True)

    # Display image and label.
    val_features, val_labels = next(iter(val_dataloader))
    print(f"Feature batch shape: {val_features.size()}")
    print(f"Labels batch shape: {len(val_labels)}")
    print([ID_TO_LABEL[label.item()] for label in val_labels])
    n = np.random.randint(64)
    img = np.transpose(val_features[n].squeeze(), axes=(1, 2, 0))
    label = val_labels[n]
    print(ID_TO_LABEL[label.item()])
    plt.imshow(img)
    plt.show()
