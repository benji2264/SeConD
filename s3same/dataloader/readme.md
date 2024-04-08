# Data Loaders

## Supported Datasets
<!-- TODO: add the links -->
- ADE20K
- CityScapes
- COCO

## Structure

```
s3same/
└── s3same/
    └── dataloader/
        └── ade20k.py
        └── cityscapes.py
        └── coco.py
Datasets/
└── ADE20K_2021_17_01/
└── CityScapes/
    └── gtFine/
    └── leftIng8bit_trainvaltest/
└── COCO/
    └── train2017/
    └── val2017/
    └── annotations/
        └── panoptic_train2017.json
        └── panoptic_val2017.json
```
