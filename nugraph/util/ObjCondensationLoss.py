import torch
from torch import Tensor

class ObjCondensationLoss(torch.nn.Module):
    def __init__(self, sb: float = 1.0, q_min: float = 0.5):
        super().__init__()
        self.sb = sb
        self.q_min = q_min

    def background_loss(self, beta: Tensor, y: Tensor) -> Tensor:
        # implement background loss here
        return 0
    
    def potential_loss(self, x: Tensor, beta: Tensor, y: Tensor) -> Tensor:
        # implement potential loss here
        return 0

    def forward(self, x: Tensor, beta: Tensor, y: Tensor) -> Tensor:
        return self.background_loss(beta, y) + self.potential_loss(x, beta, y)