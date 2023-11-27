import os
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

MIN_AREA = 600  # Min area for SAM regions
SCALING = 1.0  # Scaling factor for each region's bounding box


# def get_data_loaders(
#     dataset_train_ssl: Dataset,
#     dataset_train_kNN: Dataset,
#     dataset_test,
#     batch_size: int,
#     num_workers: int = None,
# ):
#     """Helper method to create dataloaders for ssl, kNN train and kNN test.

#     Args:
#         batch_size: Desired batch size for all dataloaders.
#     """
#     if not num_workers:
#         num_workers = os.cpu_count()

#     dataloader_train_ssl = DataLoader(
#         dataset_train_ssl,
#         batch_size=batch_size,
#         shuffle=True,
#         drop_last=True,
#         num_workers=num_workers,
#     )

#     dataloader_train_kNN = DataLoader(
#         dataset_train_kNN,
#         batch_size=batch_size,
#         shuffle=False,
#         drop_last=False,
#         num_workers=num_workers,
#     )

#     dataloader_test = DataLoader(
#         dataset_test,
#         batch_size=batch_size,
#         shuffle=False,
#         drop_last=False,
#         num_workers=num_workers,
#     )


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


class ViewSamplingDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        sam_sampling_data=None,
        global_views=2,
        local_views=0,
        global_transform=None,
        local_transform=None,
    ):
        """
        Custom dataset for image loading and view sampling.
        Different init parameters enable view sampling like SimCLR, DINO, MoCo
        or our own SAM-aided view sampling.
        Args:
            dataset_path: str, path to folder containing the training images.
            sam_sampling_data: str, path to csv file containing the SAM regions
                for each image. If None, the views will be sampled at random (like SimCLR).
            global_transform: torch.Compose or List, Optional Transform to be applied
                to the image to obtain global views. If List, then represents the transform to
                apply to each view. We should then have len(global_transform) == global_views
            local_transform: torch.Compose or List, Optional transform to be applied
                to the image to obtain local views. If List, then represents the transform to
                apply to each view. We should then have len(local_transform) == local_views
        """
        self.root_path = dataset_path
        self.global_transform = global_transform
        self.local_transform = local_transform

        self.local_views = local_views
        self.global_views = global_views
        self.scaling = SCALING
        self.regions_df = None

        if isinstance(self.global_transform, list):
            assert len(self.global_transform) == self.global_views

        if isinstance(self.local_transform, list):
            assert len(self.local_transform) == self.local_views

        # Find image files and remove prefix
        # self.imagenames = [
        #     path[len(dataset_path):].lstrip("/")
        #     for path in find_files(dataset_path)
        # ]
        self.imagenames = find_files(dataset_path)

        # Read csv containing the semantic regions of all images
        if sam_sampling_data:
            # Read regions file
            regions_df = pd.read_csv(sam_sampling_data)

            # Filter out small regions
            regions_df = regions_df[regions_df["area"] > MIN_AREA]

            # Keep largest region per object
            self.regions_df = regions_df.loc[
                regions_df.groupby(["img_id", "obj_id"])["area"].idxmax().values
            ]

    def __len__(self):
        return len(self.imagenames)

    def __getitem__(self, idx):
        # Read image
        filename = self.imagenames[idx]
        img = Image.open(filename).convert("RGB")
        img_w, img_h = img.size

        if self.regions_df is not None:
            # Retrieve image regions and sample one
            subpath = filename[len(self.root_path):].lstrip("/")
            rows = self.regions_df[self.regions_df["img_path"] == subpath]
            bbox = rows.sample(1).iloc[0]["bbox"]
            bbox = map(int, json.loads(bbox))

            # Add padding
            bbox = scale_bbox(bbox, scale=self.scaling, img_size=(img_h, img_w))
            x, y, w, h = bbox
            img = img.crop((x, y, x + w, y + h))

        # Apply global transforms
        if isinstance(self.global_transform, list):
            glob_crops = [t(img) for t in self.global_transform]
        else:
            glob_crops = [
                self.global_transform(img) if self.global_transform else img
                for _ in range(self.global_views)
            ]

        # Apply local transforms
        if isinstance(self.local_transform, list):
            loc_crops = [t(img) for t in self.local_transform]
        else:
            loc_crops = [
                self.local_transform(img) if self.local_transform else img
                for _ in range(self.local_views)
            ]

        return glob_crops + loc_crops
