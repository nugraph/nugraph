import torch
import math
from torch import Tensor
import torch.nn.functional as F

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == target.shape
        assert input.ndim == 2
        # clamp before sqrt to avoid inf gradient when prediction equals target exactly
        x = (input - target).square().sum(dim=1).clamp(min=1e-7).sqrt()
        return (x + F.softplus(-2. * x) - math.log(2.0)).mean()