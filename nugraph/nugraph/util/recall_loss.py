"""Recall loss, as described in https://arxiv.org/abs/2106.14917"""
import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.classification import MulticlassRecall

class RecallLoss(torch.nn.Module):
    """
    Recall loss function

    This module calculates the recall loss, as described in
    https://arxiv.org/abs/2106.14917.

    Args:
        num_classes: Number of true class labels
        ignore_index: True label index to ignore
        reduction: Reduction to apply to loss values
    """
    def __init__(self, num_classes: int, ignore_index: int = -1,
                 reduction: str = "sum"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.recall_metric = MulticlassRecall(
            num_classes=num_classes,
            average="none",
            ignore_index=ignore_index,
            sync_on_compute=True  # syncs TP/FN counts across GPUs before computing recall
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        RecallLoss forward pass

        Args:
            x: Predicted class labels
            y: True class labels
        """
        self.recall_metric = self.recall_metric.to(x.device)
        self.recall_metric.update(x, y)
        weight = 1 - self.recall_metric.compute()
        self.recall_metric.reset()

        ce = F.cross_entropy(x, y, reduction="none",
                             ignore_index=self.ignore_index)
        loss = weight[y] * ce

        # Sync total non-ignored count across GPUs
        n_valid = (y != self.ignore_index).sum().float()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(n_valid, op=dist.ReduceOp.SUM)
            n_valid = n_valid / dist.get_world_size()

        loss /= n_valid
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        raise ValueError(f'"{self.reduction}" is not a valid reduction.')
