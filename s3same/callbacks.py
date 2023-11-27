import os

import torch
import torch.nn as nn

from torchvision import transforms as T

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from lightly.data import LightlyDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import KNNClassifier, LinearClassifier, MetricCallback


torch.set_float32_matmul_precision("medium")


class EvalCallback(Callback):
    def __init__(
        self,
        dataset_path: str,
        eval_model: nn.Module,
        eval_name: str,
        batch_size: int = 128,
        max_epochs: int = 10,
        train_transform: T.Compose = None,
        val_transform: T.Compose = None,
        drop_last_train: bool = True,
        every_n_epochs: int = 10,
        accelerator: str = "gpu",
        **kwargs,
    ):
        """
        General callback for evaluating the model
        being trained on a different dataset or task.
        Current model is passed to `eval_model`, then
        wrapped in a Lightning Trainer and fitted on the
        dataset.
        For example, enables to perform linear probing
        or knn evaluation every nth epoch.

        Args:
            dataset_path: str, path to image dataset
            n_neigh: int, nb of neighbors for kNN
            batch_size: int, batch size
            input_size: int, input image resolution
            train_folder: str, name of train folder
                inside of the dataset path.
            val_folder: str, name of val folder
                inside of the dataset path.
        """
        super().__init__()
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.every_n_epochs = every_n_epochs
        self.eval_model = eval_model
        self.eval_name = eval_name
        self.epoch_count = 0
        self.accelerator = accelerator
        self.kwargs = kwargs

        num_workers = os.cpu_count()

        train_dataset = LightlyDataset(
            input_dir=os.path.join(dataset_path, "train"),
            transform=train_transform,
        )

        val_dataset = LightlyDataset(
            input_dir=os.path.join(dataset_path, "val"),
            transform=val_transform,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last_train,
            pin_memory=True,
            num_workers=num_workers,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers,
        )

        self.train_loader = train_dataloader
        self.val_loader = val_dataloader

    def on_validation_epoch_end(self, trainer, model):
        self.epoch_count += 1
        if self.epoch_count % self.every_n_epochs != 0:
            return

        # Instantiate eval model with kwargs
        eval_model = self.eval_model(model=model.backbone, **self.kwargs)

        # Run evaluation
        metric_callback = MetricCallback()
        eval_trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            callbacks=[metric_callback],
        )
        eval_trainer.fit(
            model=eval_model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )
        model.to("cuda")
        self.log(
            self.eval_name + "_val_top1",
            max(metric_callback.val_metrics["val_top1"]),
        )
        self.log(
            self.eval_name + "_val_top5",
            max(metric_callback.val_metrics["val_top5"]),
        )


class KNNCallback(EvalCallback):
    def __init__(
        self,
        dataset_path: str,
        num_classes: int,
        n_neigh: int = 200,
        batch_size: int = 128,
        input_size: int = 224,
        every_n_epochs: int = 10,
        accelerator: str = "gpu",
        exp_name: str = "",
    ):
        # No data augmentation for kNN
        transform = T.Compose(
            [
                T.Resize(input_size),
                T.CenterCrop(input_size),
                T.ToTensor(),
                T.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]
                ),
            ]
        )
        eval_name = "knn"
        if exp_name:
            eval_name += "_" + exp_name
        super().__init__(
            dataset_path,
            eval_model=KNNClassifier,
            eval_name=eval_name,
            batch_size=batch_size,
            max_epochs=1,
            train_transform=transform,
            val_transform=transform,
            drop_last_train=False,
            every_n_epochs=every_n_epochs,
            accelerator=accelerator,
            # Parameters for KNNClassifier
            num_classes=num_classes,
            knn_k=n_neigh,
            feature_dtype=torch.float16,
        )


class LinearProbingCallback(EvalCallback):
    def __init__(
        self,
        dataset_path: str,
        num_classes: int,
        max_epochs: int = 90,
        feature_dim: int = 384,
        freeze_model: bool = True,
        batch_size: int = 128,
        input_size: int = 224,
        every_n_epochs: int = 10,
        accelerator: str = "gpu",
        exp_name: str = "",
    ):
        # Light data augmentation for linear classifier
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(input_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]
                ),
            ]
        )
        val_transform = T.Compose(
            [
                T.Resize(input_size),
                T.CenterCrop(input_size),
                T.ToTensor(),
                T.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]
                ),
            ]
        )
        eval_name = "linear_probe"
        if exp_name:
            eval_name += "_" + exp_name
        super().__init__(
            dataset_path,
            eval_model=LinearClassifier,
            eval_name=eval_name,
            batch_size=batch_size,
            max_epochs=max_epochs,
            train_transform=train_transform,
            val_transform=val_transform,
            drop_last_train=True,
            every_n_epochs=every_n_epochs,
            accelerator=accelerator,
            # Parameters for LinearClassifier
            batch_size_per_device=batch_size,
            num_classes=num_classes,
            feature_dim=feature_dim,
            freeze_model=freeze_model,
        )
