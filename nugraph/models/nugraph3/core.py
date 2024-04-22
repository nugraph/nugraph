"""NuGraph core message-passing engine"""
import torch
from torch import nn
from torch_geometric.nn import MessagePassing, HeteroConv
from .types import T, TD

class NuGraphBlock(MessagePassing):
    """
    Standard NuGraph message-passing block
    
    This block generates attention weights for each graph edge based on both
    the source and target node features, and then applies those weights to
    the source node features in order to form messages. These messages are
    then aggregated into the target nodes using softmax aggregation, and
    then fed into a two-layer MLP to generate updated target node features.

    Args:
        source_features: Number of source node input features
        target_features: Number of target node input features
        out_features: Number of target node output features
    """
    def __init__(self, source_features: int, target_features: int,
                 out_features: int):
        super().__init__(aggr="softmax")

        self.edge_net = nn.Sequential(
            nn.Linear(source_features+target_features, source_features),
            nn.Sigmoid())

        self.net = nn.Sequential(
            nn.Linear(source_features+target_features, out_features),
            nn.Tanh(),
            nn.Linear(out_features, out_features),
            nn.Tanh())

    def forward(self, x: T, edge_index: T) -> T:
        """
        NuGraphBlock forward pass
        
        Args:
            x: Node feature tensor
            edge_index: Edge index tensor
        """
        return self.propagate(edge_index, x=x)

    def message(self, x_i: T, x_j: T) -> T:
        """
        NuGraphBlock message function

        This function constructs messages on graph edges. Features from the
        source and target nodes are concatenated and fed into a linear layer
        to construct attention weights. Messages are then formed on edges by
        weighting the source node features by these attention weights.
        
        Args:
            x_i: Edge features from target nodes
            x_j: Edge features from source nodes
        """
        return self.edge_net(torch.cat((x_i, x_j), dim=1).detach()) * x_j

    def update(self, aggr_out: T, x: T) -> T:
        """
        NuGraphBlock update function

        This function takes the output node features and combines them with
        the input features

        Args:
            aggr_out: Tensor of aggregated node features
            x: Target node features
        """
        if isinstance(x, tuple):
            _, x = x
        return self.net(torch.cat((aggr_out, x), dim=1))

class NuGraphCore(nn.Module):
    """
    NuGraph core message-passing engine
    
    This is the core NuGraph message-passing loop

    Args:
        planar_features: Number of features in planar embedding
        nexus_features: Number of features in nexus embedding
        interaction_features: Number of features in interaction embedding
        planes: List of detector planes
    """
    def __init__(self,
                 planar_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 planes: list[str]):
        super().__init__()

        full_nexus_features = len(planes) * nexus_features

        # internal planar message-passing
        self.plane_net = HeteroConv({
            (p, "plane", p): NuGraphBlock(planar_features,
                                          planar_features,
                                          planar_features)
            for p in planes})

        # message-passing from planar nodes to nexus nodes
        self.plane_to_nexus = HeteroConv({
            (p, "nexus", "sp"): NuGraphBlock(planar_features,
                                             nexus_features,
                                             nexus_features)
            for p in planes}, aggr="cat")

        # message-passing from nexus nodes to interaction nodes
        self.nexus_to_interaction = HeteroConv({
            ("sp", "in", "evt"): NuGraphBlock(full_nexus_features,
                                              interaction_features,
                                              interaction_features)})

        # message-passing from interaction nodes to nexus nodes
        self.interaction_to_nexus = HeteroConv({
            ("evt", "owns", "sp"): NuGraphBlock(interaction_features,
                                                full_nexus_features,
                                                nexus_features)})

        # message-passing from nexus nodes to planar nodes
        self.nexus_to_plane = HeteroConv({
            ("sp", "nexus", p): NuGraphBlock(nexus_features,
                                             planar_features,
                                             planar_features)
            for p in planes})

    def forward(self, p: TD, n: TD, i: TD, edges: TD) -> tuple[TD, TD, TD]:
        """
        NuGraphCore forward pass
        
        Args:
            p: Planar embedding tensor dictionary
            n: Nexus embedding tensor dictionary
            i: Interaction embedding tensor dictionary
            edges: Edge index tensor dictionary
        """
        p = self.plane_net(p, edges)
        n = self.plane_to_nexus(p|n, edges)
        i = self.nexus_to_interaction(n|i, edges)
        n = self.interaction_to_nexus(n|i, edges)
        p = self.nexus_to_plane(p|n, edges)
        return p, n, i
