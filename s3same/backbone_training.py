import os
from torch.utils.data import DataLoader

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import dataloader.cityscapes as cs
import dataloader.coco as coco

from models.build_model import (
    SupervisedClassifier,
    StudentClassifier,
    StudentContrastive,
)
from callbacks import LinearProbingCallback
from utils import get_device, get_transforms

torch.set_float32_matmul_precision("medium")

DATA_PATH = "/root/datasets/data/"

BATCH_SIZE = 512
# BATCH_SIZE = 128
# MAX_EPOCHS = 100
MAX_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4

DATASET = "coco"  # CHANGE ME! Choose dataset amongst keys in dict above
BACKBONE = "resnet18"  # CHANGE ME! One of ['vits16', 'resnet18', 'resnet50']
INPUT_SIZE = 224

# CHANGE ME! One of ['supervised', 'distillation', 'contrastive']
LEARNING_SIGNAL = "supervised"

# CHANGE ME! Path to model checkpoint. Required if `LEARNING_SIGNAL` set to "distillation"
TEACHER_CKPT = "/root/SeConD/s3same/tb_logs/coco_pretrain_resnet18_50ep_512bs/version_0/checkpoints/epoch=49-step=5700.ckpt"
# Note: here the teacher and student are assumed to be the same architecture (self-distillation)

# CHANGE ME! Number of crops. Required if `LEARNING_SIGNAL` set to "contrastive"
N_VIEWS = 4
# CHANGE ME! Scaling factor for box dimensions. Used only if `LEARNING_SIGNAL` is set to "contrastive"
SCALE_FACTOR = 2


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
    nb_views, scale_factor = (N_VIEWS, SCALE_FACTOR) if LEARNING_SIGNAL == "contrastive" else (1, None)
    dataset_builder = datasets[DATASET]
    path = os.path.join(DATA_PATH, DATASET)
    train_transforms, val_transforms = get_transforms(INPUT_SIZE)
    train_dataset = dataset_builder(
        path=path, 
        type="train", 
        transform=train_transforms,
        scale_factor=scale_factor,
        nb_views=nb_views,
    )
    val_dataset = dataset_builder(path=path, type="val", transform=val_transforms)
    num_classes = train_dataset.num_classes
    
    # Create model
    params = {
        "backbone_name": BACKBONE,
        "num_classes": num_classes,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
    }

    trainer_params = {"strategy": "auto"}
    callbacks = []
    if LEARNING_SIGNAL == "supervised":
        # model = SupervisedClassifier(**params)
        ckpt = "/root/SeConD/s3same/tb_logs/coco_contrastive_resnet18_100ep_128bs/version_5/checkpoints/epoch=99-step=45900.ckpt"
        out_ckpt = "/root/SeConD/s3same/tb_logs/coco_contrastive_resnet18_100ep_128bs/version_5/checkpoints/contrastive_ep100_renamed.ckpt"

        new_state_dict = {}
        lck = torch.load(ckpt, map_location=torch.device("cpu"))
        state_dict = lck["state_dict"]
        prefix = "projection_head"
        for name, mod in state_dict.items():
            if name.startswith(prefix):
                continue
            new_state_dict[name] = mod
        lck["state_dict"] = new_state_dict
        torch.save(lck, out_ckpt)
        
        
        model = SupervisedClassifier.load_from_checkpoint(out_ckpt, strict=False, **params)
        # project_name = "pretrain"
        project_name = "contrastive_retrain"
        run_name = f"{DATASET}_pretrain_{BACKBONE}_{MAX_EPOCHS}ep_{BATCH_SIZE}bs"
        callbacks += [
            ModelCheckpoint(monitor="val_loss")
        ]

    elif LEARNING_SIGNAL == "distillation":
        teacher = SupervisedClassifier.load_from_checkpoint(TEACHER_CKPT, **params)
        # teacher.load_state_dict(torch.load(TEACHER_CKPT))
        # teacher.to(device)
        model = StudentClassifier(teacher_model=teacher, **params)
        project_name = "distill"
        trainer_params["strategy"] = "ddp_find_unused_parameters_true"
        callbacks += [
            ModelCheckpoint(monitor="val_loss")
        ]


    elif LEARNING_SIGNAL == "contrastive":
        inference_path = f"inference_{DATASET}.csv"
        model = StudentContrastive(region_proposals=inference_path, **params)
        project_name = "contrastive"
        trainer_params["strategy"] = "ddp_find_unused_parameters_true"

        # Linear Probing callback
        linear_train_dataset = dataset_builder(path=path, type="train", transform=train_transforms)
        linear_callback = LinearProbingCallback(
            linear_train_dataset,
            val_dataset,
            num_classes=num_classes,
            feature_dim=model.embed_size,
            input_size=INPUT_SIZE,
            max_epochs=60,
            every_n_epochs=MAX_EPOCHS-1, # max_epochs - 1,  # only perform at the end
        )
        callbacks += [linear_callback, #ModelCheckpoint(monitor="linear_probe_val_top1")
                     ]

        # raise ValueError("Not implemented yet")

    # Build dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        shuffle=True,
        num_workers=10,  # os.cpu_count(),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=10,  # os.cpu_count(),
    )
    # Train
    run_name = f"{DATASET}_{project_name}_{BACKBONE}_{MAX_EPOCHS}ep_{BATCH_SIZE}bs"

    model.train_dataloader = lambda: train_dataloader
    loggers = [
        TensorBoardLogger("tb_logs", name=run_name),
        WandbLogger(project=project_name, name=run_name),
    ]

    callbacks += [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint() # save every epoch
    ]

    trainer = pl.Trainer(
        accelerator=device,
        devices="auto",
        # strategy="auto",
        # strategy="ddp_find_unused_parameters_true",
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        log_every_n_steps=5,
        logger=loggers,
        **trainer_params,
    )

    trainer.fit(model, train_dataloader, val_dataloader)
