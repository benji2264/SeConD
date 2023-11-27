from typing import Any

import torch

from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning import reducers


class BatchedNTXentLoss:
    def __init__(self, temperature, max_anchors=32):
        """
        Wrapper around pytorch-metric-learning's NTXentLoss.
        Batched implementation of the NTXent loss for lower memory requirements.
        See: https://github.com/KevinMusgrave/pytorch-metric-learning/issues/547
        All credits goes to Kevin Musgrave.
        Args:
            max_anchors: int, number of anchors for loss computation.
                if there are more anchors, we will iterate through them.
        """
        self.max_anchors = max_anchors
        self.loss_fn = NTXentLoss(temperature, reducer=reducers.DoNothingReducer())

    def __call__(self, embeddings, labels):
        """
        Loss computation.
        Args:
            embeddings: tensor of shape (batch_size * n_views, proj_head_dim),
                embeddings for all views of all images from the batch
            labels: int tensor of shape (batch_size * n_views, 1), labels of the
                embeddings. Views coming from the same image should have the
                same label, but different from all views from different images.
                e.g. labels = [1, 1, 2, 2, 3, 3, 4, 4] indicates 4 different images
                with 2 views each.
        """
        e = self.max_anchors

        all_losses = []
        for s in range(0, len(embeddings), self.max_anchors):
            # Get current anchors
            curr_anchors = embeddings[s:e]
            curr_anchor_labels = labels[s:e]

            # Manually get all pairs because we need to remove self-comparisons
            all_pairs = lmu.get_all_pairs_indices(curr_anchor_labels, labels)
            all_pairs = lmu.remove_self_comparisons(
                all_pairs, torch.arange(s, e, device=embeddings.device), len(labels)
            )

            # Compute loss
            curr_loss = self.loss_fn(
                curr_anchors, indices_tuple=all_pairs, ref_emb=embeddings
            )
            all_losses.append(curr_loss["loss"]["losses"])
            s = e
            e += self.max_anchors

        all_losses = torch.cat(all_losses, dim=0)
        loss = torch.mean(all_losses)
        return loss
