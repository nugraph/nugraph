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
        instance_features: Number of instance features
        nexus_feature: Number of nexus node features
        interaction_features: Number of interaction node features
    """
    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 instance_features: int,
                 nexus_features: int,
                 interaction_features: int):
        super().__init__()

        # store node feature sizes
        self.instance_features = instance_features
        self.nexus_features = nexus_features
        self.interaction_features = interaction_features

        # define input modules
        self.input_norm = InputNorm(in_features)
        self.planar_net = torch.nn.Linear(in_features, planar_features)
        self.beta_net = torch.nn.Linear(in_features, 1)
        self.coord_net = torch.nn.Linear(in_features, instance_features)

    def forward(self, data: NuGraphData) -> None:
        """
        NuGraph3 encoder forward pass
        
        Args:
            data: Graph data object
        """

        x_norm = self.input_norm(data["hit"].x)

        #Initializing x, of, ox explicitly
        data["hit"].x = self.planar_net(x_norm)
        data["hit"].of = self.beta_net(x_norm)
        data["hit"].ox = self.coord_net(x_norm)

        # Nexus init
        data["sp"].x = torch.zeros(data["sp"].num_nodes,
                                    self.nexus_features,
                                    device=x_norm.device)