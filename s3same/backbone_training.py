import os
from torch.utils.data import DataLoader

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import dataloader.cityscapes as cs
import dataloader.coco as coco

from models.build_model import SupervisedClassifier, StudentClassifier
from utils import get_device, get_transforms

torch.set_float32_matmul_precision("medium")

DATA_PATH = "/root/datasets/data/"

BATCH_SIZE = 512
MAX_EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4

DATASET = "coco"  # CHANGE ME! Choose dataset amongst keys in dict above
BACKBONE = "resnet18"  # CHANGE ME! One of ['vits16', 'resnet18', 'resnet50']
INPUT_SIZE = 224

# CHANGE ME! One of ['supervised', 'distillation', 'contrastive']
LEARNING_SIGNAL = "distillation"
TEACHER_CKPT = "/root/SeConD/s3same/tb_logs/coco_pretrain_resnet18_50ep_512bs/version_0/checkpoints/epoch=49-step=5700.ckpt" 
# TEACHER_CKPT = "/root/teacher_r18_coco_newname.pth"  # Path to model checkpoint. Required if `LEARNING_SIGNAL` set to "distillation"
# Note: here the teacher and student are assumed to be the same architecture (self-distillation)

if __name__ == "__main__":

    device = get_device()
    datasets = {
        # dataset_name: Dataset
        "cityscapes": cs.CityScapes,
        "coco": coco.COCO,
    }

    assert (
        DATASET in datasets
    ), f"`DATASET` should be one of {list(datasets.keys())}, got {DATASET}"
    print(f"Running on {device} device.")
    assert LEARNING_SIGNAL in [
        "supervised",
        "distillation",
        "contrastive",
    ], f"`LEARNING_SIGNAL` should be one of ['supervised', 'distillation', 'contrastive'], got {LEARNING_SIGNAL}"

    # Build datasets
    dataset_builder = datasets[DATASET]
    path = os.path.join(DATA_PATH, DATASET)
    train_transforms, val_transforms = get_transforms(INPUT_SIZE)
    train_dataset = dataset_builder(path=path, type="train", transform=train_transforms)
    val_dataset = dataset_builder(path=path, type="val", transform=val_transforms)
    num_classes = train_dataset.num_classes

    # Build dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        shuffle=True,
        num_workers=4,#os.cpu_count(),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=4,#os.cpu_count(),
    )

    # Create model
    params = {
        "backbone_name": BACKBONE,
        "num_classes": num_classes,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
    }

    if LEARNING_SIGNAL == "supervised":
        model = SupervisedClassifier(**params)
        project_name = "pretrain"
        run_name = f"{DATASET}_pretrain_{BACKBONE}_{MAX_EPOCHS}ep_{BATCH_SIZE}bs"

    elif LEARNING_SIGNAL == "distillation":
        teacher = SupervisedClassifier.load_from_checkpoint(TEACHER_CKPT, **params)
        # teacher.load_state_dict(torch.load(TEACHER_CKPT))
        # teacher.to(device)
        model = StudentClassifier(teacher_model=teacher, **params)
        project_name = "baseline_distill"
        run_name = f"DEBUG_{DATASET}_distill_{BACKBONE}_{MAX_EPOCHS}ep_{BATCH_SIZE}bs"

    elif LEARNING_SIGNAL == "contrastive":
        raise ValueError("Not implemented yet")

    # Train
    model.train_dataloader = lambda: train_dataloader
    loggers = [
        TensorBoardLogger("tb_logs", name=run_name),
        WandbLogger(project=project_name, name=run_name),
    ]

    trainer = pl.Trainer(
        accelerator=device,
        devices="auto",
        # strategy="auto",
        strategy="ddp_find_unused_parameters_true",
        max_epochs=MAX_EPOCHS,
        callbacks=[
            ModelCheckpoint(monitor="val_loss"),
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=5,
        logger=loggers,
    )

    trainer.fit(model, train_dataloader, val_dataloader)
