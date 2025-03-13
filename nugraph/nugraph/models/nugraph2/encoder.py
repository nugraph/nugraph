"""NuGraph2 encoder module"""
import torch

from .linear import ClassLinear
from ...util import InputNorm

T = torch.Tensor

class Encoder(torch.nn.Module):
    """
    NuGraph2 encoder module

    Args:
        in_features: Number of input features
        node_features: Number of planar node features
        planes: List of plane names
        classes: List of semantic class names
    """
    def __init__(self,
                 in_features: int,
                 node_features: int,
                 planes: list[str],
                 classes: list[str]):
        super().__init__()

        self.planes = planes
        self.num_classes = len(classes)

        self.net = torch.nn.ModuleDict()
        for p in planes:
            self.net[p] = torch.nn.Sequential(
                InputNorm(in_features),
                ClassLinear(in_features, node_features, self.num_classes),
                torch.nn.Tanh())

    def forward(self, x: dict[str, T]) -> dict[str, T]:
        """
        NuGraph2 encoder forward pass

        Args:
            x: Planar input tensor dictionary
        """
        ret = {}
        for p, net in self.net.items():
            ret[p] = net(x[p].unsqueeze(1).expand(-1, self.num_classes, -1))
        return ret
