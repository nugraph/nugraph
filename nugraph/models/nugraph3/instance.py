"""Convolution block for object condensation embedding"""
import torch
import torch.nn as nn

T = torch.Tensor
TD = dict[str, T]

class InstanceNet(nn.Module):
    """
    Convolution block to generate object condensation embedding tensors

    Args:
        node_features: Number of graph node features
        instance_features: Size of object condensation coordinate embedding
        planes: List containing the names of all planes
    """
    def __init__(self,
                 node_features: int,
                 instance_features: int,
                 planes: list[str]) -> None:
        """InstanceNet initializer"""
        super().__init__()

        self.net_coords = nn.ModuleDict()
        self.net_filter = nn.ModuleDict()
        for p in planes:
            self.net_filter[p] = nn.Linear(node_features, 1)
            self.net_coords[p] = nn.Linear(node_features, instance_features)

    def forward(self, x: TD, batch: TD) -> tuple[TD, TD]:
        """
        Forward pass instance net

        Args:
            x: Node embedding tensor dictionary
        """
        x_coords = {p: net(x[p]) for p, net in self.net_coords.items()}
        x_filter = {p: net(x[p]) for p, net in self.net_filter.items()}
        return x_coords, x_filter
