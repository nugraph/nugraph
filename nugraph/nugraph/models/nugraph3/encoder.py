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
    """
    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 ophit_features: int,
                 pmt_features: int,
                 flash_features: int):
        super().__init__()
        self.planar_net = nn.Linear(in_features, planar_features)
        self.nexus_features = nexus_features
        self.interaction_features = interaction_features

        # hardcode optical features pending redesign
        self.ophit_net = nn.Linear(8, ophit_features)
        self.pmt_net = nn.Linear(2, pmt_features)
        self.flash_net = nn.Linear(10, flash_features)

    def forward(self, data: Data) -> None:
        """
        NuGraph3 encoder forward pass
        
        Args:
            data: Graph data object
        """
        data["hit"].x = self.planar_net(data["hit"].x)
        data["sp"].x = torch.zeros(data["sp"].num_nodes,
                                   self.nexus_features,
                                   device=data["hit"].x.device)
        data["evt"].x = torch.zeros(data["evt"].num_nodes,
                                    self.interaction_features,
                                    device=data["hit"].x.device)
        data["ophits"].x = self.ophit_net(data["ophits"].x)
        data["opflashsumpe"].x = self.pmt_net(data["opflashsumpe"].x)
        data["opflash"].x = self.flash_net(data["opflash"].x)
