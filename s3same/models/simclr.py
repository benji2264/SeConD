import yaml
import torch

from lightly.models.modules import heads

# from lightly.utils.benchmarking import BenchmarkModule
from pytorch_lightning import LightningModule

from models.backbone import load_backbone
from losses import BatchedNTXentLoss


class SimCLRModel(LightningModule):
    def __init__(self, config, max_epochs, lr_factor=1):
        super().__init__()
        self.max_epochs = max_epochs
        self.lr_factor = lr_factor
        self.nb_global = config["global_views"]

        # Backbone
        model_name = config["backbone"]
        self.backbone, embed_size = load_backbone(model_name)
        self.embed_size = embed_size

        # Projection head
        self.projection_head = heads.SimCLRProjectionHead(
            embed_size, embed_size, config["projection_dim"]
        )
        self.criterion = BatchedNTXentLoss(config["temperature"])

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, idx):
        batch_size = len(batch[0])
        global_views = batch[: self.nb_global]
        local_views = batch[self.nb_global :]

        # Forward views
        z_global = self.forward(torch.vstack(global_views))
        z_local = self.forward(torch.vstack(local_views))
        z = torch.vstack([z_global, z_local])

        # Create labels
        labels = torch.hstack([torch.arange(batch_size)] * len(batch)).to("cuda")
        loss = self.criterion(z, labels)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * self.lr_factor, momentum=0.9, weight_decay=5e-4
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.max_epochs
        )
        return [optim], [cosine_scheduler]
