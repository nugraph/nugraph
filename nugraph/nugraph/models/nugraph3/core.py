"""NuGraph core message-passing engine"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from .types import T, Data

class NuGraphBlock(MessagePassing): # pylint: disable=abstract-method
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

        # hardcode for testing
        edge_features = 64
        attn_features = 8

        in_features = source_features + target_features + edge_features

        self.edge_net = nn.Sequential(
            nn.Linear(in_features, edge_features),
            nn.Mish(),
            nn.Linear(edge_features, edge_features),
            nn.Mish(),
            nn.Linear(edge_features, edge_features),
            nn.Mish(),
        )

        self.attn_net = nn.Sequential(
            nn.Linear(in_features, attn_features),
            nn.Mish(),
            nn.Linear(attn_features, attn_features),
            nn.Mish(),
            nn.Linear(attn_features, 1),
            nn.Sigmoid(),
        )

        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Mish(),
            nn.Linear(out_features, out_features),
            nn.Mish(),
            nn.Linear(out_features, out_features),
            nn.Mish(),
        )

    def forward(self, x: T, edge_index: T, x_e: T) -> T: # pylint: disable=arguments-differ
        """
        NuGraphBlock forward pass
        
        Args:
            x: Node feature tensor
            edge_index: Edge index tensor
            x_e: Edge feature tensor
        """
        return self.propagate(edge_index, x=x, x_e=x_e)

    def message(self, x_i: T, x_j: T, x_e: T) -> T: # pylint: disable=arguments-differ
        """
        NuGraphBlock message function

        This function constructs messages on graph edges. Features from the
        source and target nodes are concatenated and fed into a linear layer
        to construct attention weights. Messages are then formed on edges by
        weighting the source node features by these attention weights.
        
        Args:
            x_i: Edge features from target nodes
            x_j: Edge features from source nodes
            x_e: Persistent edge features
        """
        x_e = self.edge_net(torch.cat((x_i, x_j, x_e), dim=1))
        a = self.attn_net(torch.cat((x_i, x_j, x_e), dim=1))
        return a * torch.cat((x_j, x_e), dim=1)

    def update(self, aggr_out: T, x: T, x_e: T) -> T: # pylint: disable=arguments-differ
        """
        NuGraphBlock update function

        This function takes the output node features and combines them with
        the input features

        Args:
            aggr_out: Tensor of aggregated node features
            x: Target node features
            x_e: Persistent edge features
        """
        _, x = x
        return self.net(torch.cat((aggr_out, x), dim=1)), x_e

class NuGraphCore(nn.Module):
    """
    NuGraph core message-passing engine
    
    This is the core NuGraph message-passing loop

    Args:
        hit_features: Number of features in planar embedding
        nexus_features: Number of features in nexus embedding
        interaction_features: Number of features in interaction embedding
        use_checkpointing: Whether to use checkpointing
    """
    def __init__(self,
                 hit_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 use_checkpointing: bool = True):
        super().__init__()

        self.use_checkpointing = use_checkpointing

        # internal planar message-passing
        self.plane_net = NuGraphBlock(hit_features, hit_features,
                                      hit_features)

        # message-passing from planar nodes to nexus nodes
        self.plane_to_nexus = NuGraphBlock(hit_features, nexus_features,
                                           nexus_features)

        # message-passing from nexus nodes to interaction nodes
        self.nexus_to_interaction = NuGraphBlock(nexus_features,
                                                 interaction_features,
                                                 interaction_features)

        # message-passing from interaction nodes to nexus nodes
        self.interaction_to_nexus = NuGraphBlock(interaction_features,
                                                 nexus_features,
                                                 nexus_features)

        # message-passing from nexus nodes to planar nodes
        self.nexus_to_plane = NuGraphBlock(nexus_features, hit_features,
                                           hit_features)

    def message_pass(self, net: nn.Module, data: Data,
                     source: str, edge: str, target: str) -> None:
        """
        Pass messages between graph nodes, using checkpointing if enabled.

        Args:
            net: Network module
            data: Graph data object
            source: Name of source node type
            edge: Name of edge type
            target: Name of target node type
        """

        x = (data[source].x, data[target].x)

        # if no edge store, reverse the direction
        e = (source, edge, target)
        if e not in data.edge_types:
            e = (target, edge, source)
            e_idx = data[e].edge_index[(1, 0), :]
        else:
            e_idx = data[e].edge_index
        x_e = data[e].x

        # run model, using checkpoint if enabled
        if self.use_checkpointing and self.training:
            x, x_e = checkpoint(net, x, e_idx, x_e, use_reentrant=False)
        else:
            x, x_e = net(x, e_idx, x_e)

        # update embeddings
        data[target].x = x
        data[e].x = x_e

    def forward(self, data: Data) -> None:
        """
        NuGraphCore forward pass
        
        Args:
            data: Graph data object
        """

        self.message_pass(self.plane_net, data, "hit", "delaunay-planar", "hit")
        self.message_pass(self.plane_to_nexus, data, "hit", "nexus", "sp")
        self.message_pass(self.nexus_to_interaction, data, "sp", "in", "evt")
        self.message_pass(self.interaction_to_nexus, data, "evt", "in", "sp")
        self.message_pass(self.nexus_to_plane, data, "sp", "nexus", "hit")
