from typing import Optional, List, Set
from csv import DictWriter
from pathlib import Path

import numpy as np
import cv2
from torch.nn import Module
from tqdm import tqdm

from mmdet.apis import init_detector as det_init_detector, inference_detector as det_inference_detector
from mmseg.apis import init_model as seg_init_model, inference_model as seg_inference_model
from mmseg.datasets import CityscapesDataset

from SeConD.s3same.dataloader.cityscapes import AVOID_LABELS as CITY_AVOID_LABELS

class DetInferencer():
    model: Module
    min_score: Optional[int] = None
    min_area: Optional[int] = None

    def __init__(self,
        model_cfg: str,
        model_weights: str,
        min_score: Optional[int] = None,
        min_area: Optional[int] = None,
        device: str = 'cpu',
    ) -> None:
        self.model = det_init_detector(
            config=model_cfg,
            checkpoint=model_weights,
            device=device
        )
        self.min_score = min_score
        self.min_area = min_area
    
    @staticmethod
    def get_dict_keys():
        return ["img_path", "scores", "labels", "bboxes"]
    
    def raw_infer(self, imgs: List[str]):
        results = det_inference_detector(self.model, imgs=imgs)
        return [
            {
                "img_path": e.img_path,
                "scores": e.pred_instances.scores.cpu().numpy(),
                "labels": e.pred_instances.labels.cpu().numpy(),
                "bboxes": e.pred_instances.bboxes.cpu().numpy()
            } for e in results
        ]
    
    def infer(self, imgs: List[str]):
        results = self.raw_infer(imgs)
        for r in results:
            if self.min_score:
                # Find indexes with score above treshold
                idx = np.where(r['scores'] >= self.min_score)
                r["scores"] = r["scores"][idx]
                r["labels"] = r["labels"][idx]
                r["bboxes"] = r["bboxes"][idx]

            x = r['bboxes']
            x_y_width_height = np.zeros_like(x)
            x_y_width_height[:, 0] = np.minimum(x[:, 0], x[:, 2])  # x as the smaller of x1 and x2
            x_y_width_height[:, 1] = np.minimum(x[:, 1], x[:, 3])  # y as the smaller of y1 and y2
            x_y_width_height[:, 2] = np.abs(x[:, 2] - x[:, 0])  # absolute width = |x2 - x1|
            x_y_width_height[:, 3] = np.abs(x[:, 3] - x[:, 1])  # absolute height = |y2 - y1|
            r['bboxes'] = x_y_width_height
        return results




class SegInferencer():
    def __init__(self,
        model_cfg: str,
        model_weights: str,
        min_score: int = 0,
        min_area: int = 0,
        device: str = 'cpu',
        ignored_classes: Optional[Set[int]] = None
    ) -> None:
        self.model: Module = seg_init_model(
            config=model_cfg,
            checkpoint=model_weights,
            device=device
        )
        self.min_score = min_score
        self.min_area = min_area
        self.ignored_classes = ignored_classes
    
    @staticmethod
    def get_dict_keys():
        return ["img_path", "scores", "labels", "bboxes"]
     
    def _extract_class_bb(self, cls: int, mask):
        # Create a binary mask for the current class
        class_mask = (mask == cls).astype(np.uint8)
        # Define the kernel for dilation / erosion
        kernel_size = (5, 5)  # The size of the kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)  # Rectangle-shaped kernel

        # Apply dilation
        class_mask = cv2.dilate(class_mask, kernel, iterations=1)
        class_mask = cv2.erode(class_mask, kernel, iterations=1)
        num_labels, labels = cv2.connectedComponents(class_mask)

        bounding_boxes = []
        for label in range(1, num_labels):
            # Find coordinates of pixels with this label
            y, x = np.where(labels == label)
            if y.size == 0 or x.size == 0:
                continue  # Skip if there are no pixels with this label

            # Determine the bounding box coordinates
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()

            # Store the bounding box (x_min, y_min, width, height)
            bounding_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
        return bounding_boxes

    def raw_infer(self, imgs: List[str]):
        results = seg_inference_model(self.model, imgs)
        return [
            {
                "img_path": e.img_path,
                "pred_seg": e.pred_sem_seg.data.cpu().numpy()[0, :, :]
            } for e in results
        ]
    
    def infer(self, imgs: List[str]):
        raw_results = self.raw_infer(imgs)
        results = []
        for r in raw_results:
            cur_result = {"img_path": r["img_path"], "labels": [], "bboxes": [], "scores": []}
            # Identify unique classes in the mask & filter ignored classes
            mask = r["pred_seg"]
            unique_classes = np.unique(mask)
            if self.ignored_classes:
                unique_classes = np.array([cls for cls in unique_classes if cls not in self.ignored_classes])
            for cls in unique_classes:
                bb = self._extract_class_bb(cls, mask)
                cur_result["labels"] += [cls] * len(bb)
                cur_result["bboxes"] += bb
            results.append(cur_result)

        return results

def get_ignored_idx_city(data_root: str) -> Set[int]:
    ds = CityscapesDataset(data_root=data_root)
    mapping = {cls: i for i, cls in enumerate(ds.metainfo['classes'])}
    return {i for cls, i in mapping.items() if cls in CITY_AVOID_LABELS}


def infer_city():
    CITY_PATH = "datasets/data/cityscapes"
    MODEL = "config_seg_city_r18.py"
    WEIGHTS = "models/seg_r18_city_iter_80000.pth"
    inferencer = SegInferencer(
        model_cfg=MODEL,
        model_weights=WEIGHTS,
        device='cuda',
        ignored_classes = get_ignored_idx_city('datasets/cityscapes')
    )
    city_path = Path(CITY_PATH) / 'leftImg8bit' / 'train'
    imgs = list(city_path.rglob('*.png'))
    return inferencer, imgs

def infer_coco():
    COCO_PATH = "/home/hice1/lmichalski3/scratch/datasets/data/coco/train2017"
    MODEL = "config_det_coco_r18.py"
    WEIGHTS = "models/epoch_10.pth"
    MIN_SCORE = 0.3
    coco_path = Path(COCO_PATH) / "train2017"
    imgs = list(Path(coco_path).glob("*.jpg"))

    inferencer = DetInferencer(
        model_cfg=MODEL,
        model_weights=WEIGHTS,
        device='cuda',
        min_score=MIN_SCORE
    )
    return inferencer, imgs

if __name__ == "__main__":
    # inferencer, imgs = infer_coco()
    inferencer, imgs = infer_city()

    print(f"Found {len(imgs)} images")

    batch_size = 30
    with open("inference_city.csv", mode='w') as csvfile:
        dict_keys = inferencer.get_dict_keys()
        writer = DictWriter(csvfile, fieldnames=dict_keys)
        writer.writeheader()

        for i in tqdm(range(0, len(imgs), batch_size)):
            batch = imgs[i:i + batch_size]
            for r in inferencer.infer(batch):
                # r[dict_keys[1]] = r[dict_keys[1]]
                # r[dict_keys[2]] = r[dict_keys[2]]
                # r[dict_keys[3]] = r[dict_keys[3]]
                writer.writerow(r)
            