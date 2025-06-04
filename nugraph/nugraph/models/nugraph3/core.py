"""NuGraph core message-passing engine"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from .types import T, TD, Data

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

        self.edge_net = nn.Sequential(
            nn.Linear(source_features+target_features, 1),
            nn.Sigmoid())

        self.net = nn.Sequential(
            nn.Linear(source_features+target_features, out_features),
            nn.Mish(),
            nn.Linear(out_features, out_features),
            nn.Mish())

    def forward(self, x: T, edge_index: T) -> T: # pylint: disable=arguments-differ
        """
        NuGraphBlock forward pass
        
        Args:
            x: Node feature tensor
            edge_index: Edge index tensor
        """
        return self.propagate(edge_index, x=x)

    def message(self, x_i: T, x_j: T) -> T: # pylint: disable=arguments-differ
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

    def update(self, aggr_out: T, x: T) -> T: # pylint: disable=arguments-differ
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

        # message-passing between nexus nodes and PMT nodes (opflashsumpe)
        self.nexus_to_pmt = NuGraphBlock(nexus_features, pmt_features, pmt_features)
        self.pmt_to_nexus = NuGraphBlock(pmt_features, nexus_features, nexus_features)

    def checkpoint(self, net: nn.Module, *args) -> TD:
        """
        Checkpoint module, if enabled.
        
        Args:
            net: Network module
            args: Arguments to network module
        """
        if self.use_checkpointing and self.training:
            return checkpoint(net, *args, use_reentrant=False)
        else:
            return net(*args)

    def forward(self, data: Data) -> None:
        """
        NuGraphCore forward pass
        
        Args:
            data: Graph data object
        """

        # message-passing in hits
        data["hit"].x = self.checkpoint(
            self.plane_net, data["hit"].x,
            data["hit", "delaunay-planar", "hit"].edge_index)

        # message-passing from hits to nexus
        data["sp"].x = self.checkpoint(
            self.plane_to_nexus, (data["hit"].x, data["sp"].x),
            data["hit", "nexus", "sp"].edge_index)

        # message-passing from nexus to interaction
        data["evt"].x = self.checkpoint(
            self.nexus_to_interaction, (data["sp"].x, data["evt"].x),
            data["sp", "in", "evt"].edge_index)

        # message-passing from interaction to nexus
        data["sp"].x = self.checkpoint(
            self.interaction_to_nexus, (data["evt"].x, data["sp"].x),
            data["sp", "in", "evt"].edge_index[(1,0), :])

        # message-passing from nexus to hits
        data["hit"].x = self.checkpoint(
            self.nexus_to_plane, (data["sp"].x, data["hit"].x),
            data["hit", "nexus", "sp"].edge_index[(1,0), :])

        # for connections between opflashsumpe and nexus
        if ("opflashsumpe" in data.node_types and ("sp", "connection", "opflashsumpe") in data.edge_types):
            # message-passing from space points to PMTs
            data["opflashsumpe"].x = self.checkpoint(
                self.nexus_to_pmt, (data["sp"].x, data["opflashsumpe"].x),
                data["sp", "connection", "opflashsumpe"].edge_index[(1,0), :])

            # message-passing from PMTs to space points
            data["sp"].x = self.checkpoint(
                self.pmt_to_nexus, (data["opflashsumpe"].x, data["sp"].x),
                data["opflashsumpe", "connection", "sp"].edge_index[(1,0), :])
