from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import torchmetrics

from pytorch_lightning import LightningModule
from lightly.models.modules.heads import SimCLRProjectionHead as ProjectionHead
from lightly.transforms.dino_transform import DINOTransform

from models.backbone import load_backbone
from losses import BatchedNTXentLoss


class GenericModel(LightningModule):
    def __init__(self, backbone_name, learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        self.backbone, self.embed_size = load_backbone(backbone_name)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.metrics = {}
        self.backbone.train()

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
        optimizer = Adam(
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
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )

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
        self.log(
            "val_acc_step", self.accuracy, on_step=True, on_epoch=True, sync_dist=True
        )

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
        x, _ = batch
        logits = self(x)
        
        with torch.no_grad():
            pseudo_y = self.teacher_model(x).argmax(axis=-1)
                
        loss = self.criterion(logits, pseudo_y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # Loss with real labels
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, sync_dist=True)

        # Loss with teacher pseudo labels
        with torch.no_grad():
            pseudo_y = self.teacher_model(x).argmax(axis=-1)
            
        loss = self.criterion(logits, pseudo_y)
        self.log("val_loss_teacher", loss, sync_dist=True)

        # Accuracy with real labels
        self.accuracy(logits, y)
        self.log('val_acc_step', self.accuracy, on_step=True, on_epoch=True, sync_dist=True)

        # Accuracy with teacher pseudo labels
        self.accuracy(logits, pseudo_y)
        self.log('val_acc_step_teacher', self.accuracy, on_step=True, on_epoch=True, sync_dist=True)


class StudentContrastive(LightningModule):

    def __init__(
        self,
        backbone_name,
        num_classes,
        region_proposals,
        learning_rate=1e-3,
        weight_decay=1e-4,
        projection_dim=128,
        temperature=0.07,
    ):
        super().__init__()
        self.backbone, self.embed_size = load_backbone(backbone_name)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.region_proposals = region_proposals
        self.flatten = nn.Flatten()
        self.projection_head = self.projection_head = ProjectionHead(
            self.embed_size, self.embed_size, projection_dim
        )
        self.temperature = temperature
        self.criterion = BatchedNTXentLoss(self.temperature, max_anchors=32)
        # self.log("linear_probe_val_top1", 0)

    # def on_train_epoch_start(self):
    #     self.backbone.train()
        
    def forward(self, x):
        x = self.backbone(x)
        features = self.flatten(x)
        proj = self.projection_head(features)
        return proj

    def training_step(self, batch, batch_idx):
        x, _ = batch
        n_views, batch_size = len(x), len(x[0])

        # Stack all views [(bs, c, h, w)] * n_views -> [bs * n_views, c, h, w]
        x = torch.vstack(x)
        logits = self(x)
        y = torch.hstack([torch.arange(batch_size)] * n_views).to(self.device)

        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # define val step for linear callback
        pass
    
    def configure_optimizers(self):
        optim = SGD(self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4)
        cosine_scheduler = CosineAnnealingLR(optim, self.trainer.max_epochs)
        return [optim], [cosine_scheduler]
