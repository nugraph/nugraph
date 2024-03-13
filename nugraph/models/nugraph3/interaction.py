"""Convolution block for interaction hierarchy level"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from torch_geometric.nn.aggr import AttentionalAggregation

T = torch.Tensor
TD = dict[str, T]

class InteractionNet(nn.Module):
    """
    This convolution block projects graph node features from each plane up to
    a global embedding representing the interaction as a whole.

    Args:
        planar_features: Number of features in the planar embedding
        interaction_features: Number of features in the interaction embedding
        planes: List containing the names of all planes
    """
    def __init__(self,
                 planar_features: int,
                 interaction_features: int,
                 planes: list[str]) -> None:
        """InteractionNet initializer"""
        super().__init__()

        self.aggr = nn.ModuleDict()
        for p in planes:
            self.aggr[p] = AttentionalAggregation(nn.Sequential(
                nn.Linear(planar_features, planar_features),
                nn.Sigmoid(),
            ))
        
        self.net = nn.Linear(len(planes) * planar_features, interaction_features)

    def forward(self, x: TD, index: TD) -> T:
        """Forward pass for InteractionNet 
        
        Args:
            x: Dictionary containing node feature tensors for each plane
            index: Dictionary containing node index mapping for each plane
        """
        x = [ self.aggr(x[p], index[p]) for p, net in self.aggr.items() ]
        for p in self.aggr: print(p, x[p].shape)
        return self.net(torch.cat(x, dim=1))
