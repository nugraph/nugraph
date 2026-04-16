# as described in https://arxiv.org/abs/2106.14917

import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.classification import MulticlassRecall

class RecallLoss(torch.nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index
        self.recall_metric = MulticlassRecall(
            num_classes=num_classes,
            average='none',
            ignore_index=ignore_index,
            sync_on_compute=True  # syncs TP/FN counts across GPUs before computing recall
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.clone()
        target[target > 4] = -1

        self.recall_metric = self.recall_metric.to(input.device)
        self.recall_metric.update(input, target)
        weight = 1 - self.recall_metric.compute()
        self.recall_metric.reset()

        ce = F.cross_entropy(input, target, reduction='none',
                             ignore_index=self.ignore_index)
        mask = target != self.ignore_index
        loss = torch.where(mask, weight[target.clamp(min=0)] * ce,
                           torch.zeros_like(ce))

        # Sync total non-ignored count across GPUs
        n_valid = mask.sum().float()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(n_valid, op=dist.ReduceOp.SUM)
            n_valid = n_valid / dist.get_world_size()

        return loss[mask].sum() / n_valid

    def reset(self):
        self.recall_metric.reset()
