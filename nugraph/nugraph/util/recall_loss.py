"""Recall loss, as described in https://arxiv.org/abs/2106.14917"""
import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn.functional as F

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

    @torch.jit.unused
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        RecallLoss forward pass

        Args:
            x: Predicted class labels
            y: True class labels
        """
        # per-class recall from counts, without the boolean selections of
        # MulticlassRecall that make the CPU wait on the GPU; classes absent
        # from the batch get recall 0, as in MulticlassRecall
        valid = y != self.ignore_index
        cls = torch.where(valid, y, 0)
        tp = torch.zeros(x.size(1), device=x.device).index_add_(
            0, cls, ((x.argmax(dim=1) == y) & valid).float())
        total = torch.zeros(x.size(1), device=x.device).index_add_(0, cls, valid.float())
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        weight = (1 - tp / total.clamp(min=1)).to(dtype=x.dtype)

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
