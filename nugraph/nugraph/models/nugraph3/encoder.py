"""NuGraph3 encoder"""
from pynuml.data import NuGraphData
from torch import nn, Tensor
from torch import Tensor
from torch.nn import Module, Sequential, Linear, Mish
from torch_geometric.nn import MessagePassing
from ...util import InputNorm

T = Tensor

class EncoderModule(MessagePassing):
    """
    Standard encoder message-passing block

    Args:
        source_features: 
    """
    def __init__(self, source_features: int, node_features: int,
                 edge_features: int):
        super().__init__(aggr="softmax")
    
        self.edge_net = Sequential(
            Linear(source_features, edge_features),
            Mish(),
            Linear(edge_features, edge_features),
            Mish(),
            Linear(edge_features, edge_features),
            Mish(),
        )

        self.node_net = Sequential(
            Linear(edge_features, node_features),
            Mish(),
            Linear(node_features, node_features),
            Mish(),
            Linear(node_features, node_features),
            Mish(),
        )   

    def forward(self, x: T, edge_index: T) -> T:
        j, i = edge_index
        msg = self.message(x[j])
        aggr_out = self.aggregate(msg, i)
        x = self.update(aggr_out)
        return x, msg

    def message(self, x_j: T):
        return self.edge_net(x_j)

    def update(self, aggr_out: T):
        return self.node_net(aggr_out)

class Encoder(Module):
    """
    NuGraph3 encoder
    
    Args:
        in_features: Number of input node features
        hit_features: Number of hit node features
        nexus_features: Number of nexus node features
        interaction_features: Number of interaction node features
        edge_features: Number of edge features
    """
    def __init__(self, in_features: int, hit_features: int,
                 nexus_features: int, interaction_features: int,
                 edge_features: int):
        super().__init__()

        self.input_norm = InputNorm(in_features)
        self.hit_net = EncoderModule(in_features, hit_features, edge_features)
        self.nexus_net = EncoderModule(hit_features, nexus_features, edge_features)
        self.interaction_net = EncoderModule(nexus_features, interaction_features,
                                            edge_features)

    def pass_message(self, net, data, source, target, edge):
        data[target].x, data[source, edge, target].x = net(
            data[source].x, data[source, edge, target].edge_index)

    def forward(self, data: NuGraphData) -> None:
        """
        NuGraph3 encoder forward pass
        
        Args:
            data: Graph data object
        """
        data["hit"].x = self.input_norm(data["hit"].x)
        self.pass_message(self.hit_net, data, "hit", "hit", "delaunay-planar")
        self.pass_message(self.nexus_net, data, "hit", "sp", "nexus")
        self.pass_message(self.interaction_net, data, "sp", "evt", "in")
