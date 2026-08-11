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
                 interaction_features: int,
                 instance_features: int,
                 edge_features_scale: float = 0.0,
                 identity_edge_update_net: bool = False):
        super().__init__()

        self.input_norm = InputNorm(in_features)
        self.planar_net = torch.nn.Linear(in_features, planar_features)

        # object condensation beta encoder
        self.beta_net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 1),
            torch.nn.Sigmoid(),
        )

        # object condensation coordinate encoder
        self.coord_net = torch.nn.Sequential(
            torch.nn.Linear(in_features, instance_features),
            torch.nn.Mish(),
        )

        self.nexus_features = nexus_features
        self.interaction_features = interaction_features
        # pre-compute edge latent dimensions per edge type to match NuGraphBlock;
        # identity_edge_update_net forces edge_features=1 regardless of scale
        if identity_edge_update_net:
            self.edge_features_pp = 1 if edge_features_scale > 0.0 else 0
            self.edge_features_pn = 1 if edge_features_scale > 0.0 else 0
            self.edge_features_ni = 1 if edge_features_scale > 0.0 else 0
        else:
            self.edge_features_pp = int(edge_features_scale * planar_features)
            self.edge_features_pn = int(edge_features_scale * min(planar_features, nexus_features))
            self.edge_features_ni = int(edge_features_scale * min(nexus_features, interaction_features))

    def forward(self, data: NuGraphData) -> None:
        """
        NuGraph3 encoder forward pass
        
        Args:
            data: Graph data object
        """
        x_in = self.input_norm(data["hit"].x)
        data["hit"].x = self.planar_net(x_in)
        data["hit"].of = self.beta_net(x_in)
        data["hit"].ox = self.coord_net(x_in)
        data["sp"].x = torch.zeros(data["sp"].num_nodes,
                                   self.nexus_features,
                                   device=data["hit"].x.device)
        data["evt"].x = torch.zeros(data["evt"].num_nodes,
                                    self.interaction_features,
                                    device=data["hit"].x.device)
        dev = data["hit"].x.device
        if self.edge_features_pp > 0:
            e = data["hit", "delaunay-planar", "hit"]
            e.edge_attr = torch.zeros(e.edge_index.shape[1], self.edge_features_pp, device=dev)
        if self.edge_features_pn > 0:
            e = data["hit", "nexus", "sp"]
            e.edge_attr_fwd = torch.zeros(e.edge_index.shape[1], self.edge_features_pn, device=dev)
            e.edge_attr_bwd = torch.zeros(e.edge_index.shape[1], self.edge_features_pn, device=dev)
        if self.edge_features_ni > 0:
            e = data["sp", "in", "evt"]
            e.edge_attr_fwd = torch.zeros(e.edge_index.shape[1], self.edge_features_ni, device=dev)
            e.edge_attr_bwd = torch.zeros(e.edge_index.shape[1], self.edge_features_ni, device=dev)