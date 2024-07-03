from torch import Tensor
import torch.nn as nn

from .linear import ClassLinear

class Encoder(nn.Module):
    def __init__(self,
                 in_features: int,
                 node_features: int,
                 planes: list[str],
                 classes: list[str]):
        super().__init__()

        self.planes = planes
        self.num_classes = len(classes)

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Sequential(
                ClassLinear(in_features, node_features, self.num_classes),
                nn.Tanh())

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        return { p: net(x[p].unsqueeze(1).expand(-1, self.num_classes, -1)) for p, net in self.net.items() }