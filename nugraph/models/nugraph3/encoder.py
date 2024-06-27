"""NuGraph3 encoder"""
import torch
from torch import nn
from .types import Data

class Encoder(nn.Module):
    """
    NuGraph3 encoder
    
    Args:
        in_features: Number of input node features
        planar_features: Number of planar node features
        nexus_feature: Number of nexus node features
        interaction_features: Number of interaction node features
        planes: Tuple of planes
    """
    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 planes: tuple[str]):
        super().__init__()
        self.planar_net = nn.Linear(in_features, planar_features)
        self.nexus_net = nn.Linear(3, nexus_features) # 3D position
        self.interaction_features = interaction_features
        self.planes = planes

    def forward(self, data: Data) -> None:
        """
        NuGraph3 encoder forward pass
        
        Args:
            data: Graph data object
        """
        for p in self.planes:
            data[p].x = self.planar_net(data[p].x)
        data["sp"].x = self.nexus_net(data["sp"].pos)
        data["evt"].x = torch.zeros(data["evt"].num_nodes,
                                    self.interaction_features,
                                    device=data["sp"].x.device)
