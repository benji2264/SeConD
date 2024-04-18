from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
import torchmetrics

from models.backbone import load_backbone


class GenericModel(LightningModule):
    def __init__(self, backbone_name, learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        self.backbone, self.embed_size = load_backbone(backbone_name)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.metrics = {}

    @abstractmethod
    def forward(self, x):
        pass
    
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
        self.log("val_loss", loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                steps_per_epoch=int(
                    len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
                ),
                epochs=self.trainer.max_epochs,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


class SupervisedClassifier(GenericModel):

    def __init__(
        self, backbone_name, num_classes, learning_rate=1e-3, weight_decay=1e-4
    ):
        super().__init__(backbone_name, learning_rate, weight_decay)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.embed_size, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)


    def forward(self, x):
        x = self.backbone(x)
        features = self.flatten(x)
        out = self.linear(features)
        return out

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, sync_dist=True)
        
        self.accuracy(logits, y)
        self.log('val_acc_step', self.accuracy, on_step=True, on_epoch=True, sync_dist=True)

class StudentClassifier(SupervisedClassifier):

    def __init__(
        self,
        backbone_name,
        num_classes,
        teacher_model,
        learning_rate=1e-3,
        weight_decay=1e-4,
    ):
        super().__init__(backbone_name, num_classes, learning_rate, weight_decay)
        self.teacher_model = teacher_model
        self.teacher_model.eval()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # print(y.detach().cpu().)
        with torch.no_grad():
            pseudo_y = self.teacher_model(x)

        # loss = self.criterion(logits, pseudo_y)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss
