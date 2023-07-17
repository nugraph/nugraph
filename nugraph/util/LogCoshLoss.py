import torch
import math
from torch import Tensor
import torch.nn.functional as F

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        x = input - target
        return (x + F.softplus(-2. * x) - math.log(2.0)).mean()