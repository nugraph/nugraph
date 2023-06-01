# as described in https://arxiv.org/abs/2106.14917

import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_recall

class RecallLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        weight = 1 - multiclass_recall(input, target, num_classes=input.size(1), average='none')
        CE = F.cross_entropy(input, target, reduction='none')
        loss =  weight[target] * CE
        return loss.mean()

