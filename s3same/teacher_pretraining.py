import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision.models import resnet18
from torch.optim.lr_scheduler import OneCycleLR
import torchmetrics
import dataloader.cityscapes as cs
import dataloader.coco as coco

data_path = "/Users/benjaminmissaoui/Desktop/gt_s24/6476/s3same/s3same/datasets/"

BATCH_SIZE = 512
MAX_EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# datasets = {
#     # dataset: [/path/to/train, /path/to/val],
#     "cityscapes": [
#         "cityscapes"
#     ]
# }

datasets = {
    # dataset_name: [Dataset, num_classes]
    "cityscapes": [cs.CityScapes, 35],
    "coco": [coco.COCO, 200],
}

dataset = "coco"  # CHANGE ME! Choose dataset amongst keys in dict above


class ResNet18Module(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        self.model = resnet18(pretrained=False, num_classes=num_classes)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss)

        self.accuracy(logits, y)
        self.log("val_acc_step", self.accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                steps_per_epoch=int(
                    len(train_dataloader) / self.trainer.accumulate_grad_batches
                ),
                epochs=self.trainer.max_epochs,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


if __name__ == "__main__":
    train_transforms = v2.Compose(
        [
            v2.RandomResizedCrop(224),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Normalization without augmentation for validation data
    val_transforms = v2.Compose(
        [
            v2.Resize(224),
            v2.CenterCrop(224),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset_builder, num_classes = datasets[dataset]
    path = os.path.join(data_path, dataset)
    train_dataset = dataset_builder(path=path, type="train", transform=train_transforms)
    val_dataset = dataset_builder(path=path, type="val", transform=val_transforms)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # Replace num_classes with the number of classes in your dataset
    # num_classes = cs.NUM_CLASSES - len(cs.AVOID_LABELS)
    model = ResNet18Module(
        num_classes=num_classes, learning_rate=LR, weight_decay=WEIGHT_DECAY
    )

    print(f"Running on {device}")
    logger = TensorBoardLogger("tb_logs", name=f"{dataset}_r18_teacher_cs_v1")

    trainer = pl.Trainer(
        accelerator=device,
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            ModelCheckpoint(monitor="val_loss"),
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=5,
        logger=logger,
    )

    trainer.fit(model, train_dataloader, val_dataloader)
