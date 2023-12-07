# S3SAMe: SAM-aided view sampling for contrastive learning

![Main Figure](assets/s3same_figure.png?raw=true "Title")

We introduce **S3SAMe**, a method for sampling the views when performing contrastive learning of image features. We use SAM to first segment the input image into semantic regions, then sample the views inside of the same region. This enforces the views to represent the same object, and thus enables self-supervised learning from scene-centric datasets (e.g. Cityscapes, ADE20K, ...) as opposed to object-centric datasets (e.g. Imagenet, ...). S3SAMe can be readily added to most methods like SimCLR, DINO, MoCo, etc...

## Installation

First clone and install this repo 
```
git clone https://github.com/benji2264/s3same.git
cd s3same/s3same
pip install -e .
```

Create a conda environment.
```
conda create -n s3same python=3.8
conda activate s3same
```

## Train/Evaluate on your own datasets

### Validation sets

We log the KNN accuracy (every 10 epochs) and the linear probing accuracy (once, at the end of training). By default, evaluation is performed on 3 small classification datasets: ImageNette, cifar10 and STL-10. You can choose [here](https://github.com/benji2264/s3same/blob/45a222c1b3f1d30879cd63c44edefc3bf724a715/s3same/train.py#L32) the datasets used for evaluation, inside of ```train.py```. If you want to evaluate with these 3 default datasets, or if you want to use your own, download the dataset inside of ```s3same/data```.

The directory tree should then look like this:

```
s3same/
└── data/
    └── imagenette/
        ├── train/
        │   ├── class1/
        │   │   ├── image1.png
        │   │   └── ...
        │   ├── class2/
        │   │   ├── image1.png
        │   │   └── ...
        │   └── ...
        └── val/
            ├── class1/
            │   ├── image1.png
            │   └── ...
            ├── class2/
            │   ├── image1.png
            │   └── ...
            └── ...
```
Make sure that to use "train" and "val" for the subfolders' names. 

You now have to edit the validation sets inside of ```train.py``` with the correct path to the new dataset and the number of classes:

```python
val_sets = {
    # name: [root_path, num_classes],
    "imagenette": ["data/imagenette", 10],
}
```

### Training sets

During training, we rely on HQ-SAM to segment the input images and sample the views within the same region. This process can slow down training, we thus suggest to generate and save the segmentation masks for the whole dataset before you start training, in order to save a substantial amount of time for during the view sampling process.


## Reproduce the papers results







