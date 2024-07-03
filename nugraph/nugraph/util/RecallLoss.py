# as described in https://arxiv.org/abs/2106.14917

import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.functional import recall

class RecallLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        weight = 1 - recall(input, target, 'multiclass',
                            num_classes=input.size(1),
                            average='none',
                            ignore_index=self.ignore_index)
        ce = F.cross_entropy(input, target, reduction='none',
                             ignore_index=self.ignore_index)
        loss = weight[target] * ce
        return loss.mean()