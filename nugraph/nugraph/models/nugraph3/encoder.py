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
        nexus_features: Number of nexus node features
        interaction_features: Number of interaction node features
        instance_features: Number of instance node features
        input_edge_geom: If True, compute 5 fixed geometric features on hit-hit edges
    """
    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 instance_features: int,
                 input_edge_geom: bool = False):
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
        self.input_edge_geom = input_edge_geom

    def forward(self, data: NuGraphData) -> None:
        """
        NuGraph3 encoder forward pass

        Args:
            data: Graph data object
        """
        x_in = self.input_norm(data["hit"].x)

        if self.input_edge_geom:
            # Compute fixed geometric edge features from normalized hit inputs.
            # Recomputed each forward pass since hit features differ per batch.
            pp = data["hit", "delaunay-planar", "hit"]
            src, dst = pp.edge_index
            d_wire     = x_in[src, 0] - x_in[dst, 0]
            d_time     = x_in[src, 1] - x_in[dst, 1]
            d_integral = x_in[src, 2] - x_in[dst, 2]
            d_rms      = x_in[src, 3] - x_in[dst, 3]
            distance   = torch.hypot(d_wire, d_time)
            pp.edge_geom = torch.stack([d_integral, d_rms, d_wire, d_time, distance], dim=1)

        data["hit"].x = self.planar_net(x_in)
        data["hit"].of = self.beta_net(x_in)
        data["hit"].ox = self.coord_net(x_in)
        data["sp"].x = torch.zeros(data["sp"].num_nodes,
                                   self.nexus_features,
                                   device=data["hit"].x.device)
        data["evt"].x = torch.zeros(data["evt"].num_nodes,
                                    self.interaction_features,
                                    device=data["hit"].x.device)
