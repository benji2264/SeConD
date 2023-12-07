import os
import yaml
import argparse

import torch
import pytorch_lightning as pl

from models import simclr
from callbacks import KNNCallback, LinearProbingCallback
from datasets import ViewSamplingDataset
from transforms import model_transforms

torch.set_float32_matmul_precision("medium")

devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
accel = "gpu"

BATCH_SIZE = 256

SEED = 0
NUM_WORKERS = 24
LOGS_DIR = "./logs"
LR_FACTOR = BATCH_SIZE / 256
VAL_INPUT_SIZE = 128

models = {
    "SimCLR": simclr.SimCLRModel,
}

data_path = "./data"

val_sets = {
    # name: [root_path, num_classes],
    "imagenette": [os.path.join(data_path, "imagenette"), 10],
    "stl10": [os.path.join(data_path, "stl10"), 10],
    "cifar10": [os.path.join(data_path, "cifar10"), 10],
}


def build_model(config, max_epochs, lr_factor=1):
    model_name = config["name"]
    model = models[model_name](config, max_epochs, lr_factor)
    return model


def make_name(model_name, pretrain_name, batch_size, sam_sampling=True):
    basename = f"{model_name}_{pretrain_name}_{batch_size}bs"
    if sam_sampling:
        basename += "_sam"
    return basename


if __name__ == "__main__":
    pl.seed_everything(SEED)

    # Read args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model config file"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to dataset config file"
    )
    args = parser.parse_args()

    # Parse configs
    with open(args.model, "r") as config_file:
        model_config = yaml.safe_load(config_file)

    with open(args.data, "r") as config_file:
        dataset_config = yaml.safe_load(config_file)

    max_epochs = dataset_config["max_epochs"]
    sam_sampling = model_config.get("sam_sampling", False)

    # Build model
    model = build_model(model_config, max_epochs, LR_FACTOR)
    transforms = model_transforms[model_config["name"]]
    embed_size = model.embed_size

    # Build dataset and dataloader
    ssl_train_path = os.path.join(dataset_config["root"], dataset_config["train"])
    sampling_data = False
    if sam_sampling:
        sampling_data = os.path.join(
            dataset_config["root"], dataset_config["sam_sampling"]
        )

    pretrain_dataset = ViewSamplingDataset(
        ssl_train_path,
        sampling_data,
        model_config["global_views"],
        model_config["local_views"],
        *transforms,
    )

    pretrain_dataloader = torch.utils.data.DataLoader(
        pretrain_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    # Evaluation callbacks
    knn_callbacks = [
        KNNCallback(
            val_sets[name][0],
            num_classes=val_sets[name][1],
            input_size=VAL_INPUT_SIZE,
            every_n_epochs=10,
            exp_name=name,
        )
        for name in val_sets
    ]

    linear_callbacks = [
        LinearProbingCallback(
            val_sets[name][0],
            num_classes=val_sets[name][1],
            feature_dim=embed_size,
            input_size=VAL_INPUT_SIZE,
            max_epochs=90,
            every_n_epochs=max_epochs - 1,  # only perform at the end
            exp_name=name,
        )
        for name in val_sets
    ]

    exp_name = make_name(
        model_config["name"], dataset_config["name"], BATCH_SIZE, sam_sampling
    )
    # sub_dir = dataset_config["name"]
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=LOGS_DIR, name="", sub_dir=exp_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints")
    )

    # Training
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        accelerator=accel,
        default_root_dir=LOGS_DIR,
        logger=logger,
        callbacks=[checkpoint_callback] + knn_callbacks + linear_callbacks,
        log_every_n_steps=5,
    )

    trainer.fit(model, train_dataloaders=pretrain_dataloader)
