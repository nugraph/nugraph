"""NuGraph3 encoder"""
import torch
from pynuml.data import NuGraphData
from ...util import InputNorm

class Encoder(torch.nn.Module):
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
                 interaction_features: int):
        super().__init__()
        self.input_norm = InputNorm(in_features)
        self.planar_net = torch.nn.Linear(in_features, planar_features)
        self.nexus_features = nexus_features
        self.interaction_features = interaction_features

    def forward(self, data: NuGraphData) -> None:
        """
        NuGraph3 encoder forward pass
        
        Args:
            data: Graph data object
        """
        data["hit"].x = self.input_norm(data["hit"].x)
        data["hit"].x = self.planar_net(data["hit"].x)
        data["sp"].x = torch.zeros(data["sp"].num_nodes,
                                   self.nexus_features,
                                   device=data["hit"].x.device)
        data["evt"].x = torch.zeros(data["evt"].num_nodes,
                                    self.interaction_features,
                                    device=data["hit"].x.device)
