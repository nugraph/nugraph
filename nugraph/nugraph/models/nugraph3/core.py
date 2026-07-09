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
                 instance_features: int,
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

        # widen MLP for instance embedding generation
        hidden = 3 * hit_features

        # deeper, wider object condensation beta embedding
        self.beta_net = nn.Sequential(
            nn.Linear(hit_features + 1, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        # deeper, wider object condensation coordinate embedding
        self.coord_net = nn.Sequential(
            nn.Linear(hit_features + instance_features, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, instance_features),
        )

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

        # define quick aliases for node stores
        h, sp, evt = data["hit"], data["sp"], data["evt"]

        # message-passing in hits
        h.x = self.checkpoint(
            self.plane_net, h.x,
            data["hit", "delaunay-planar", "hit"].edge_index)

        # message-passing from hits to nexus
        sp.x = self.checkpoint(
            self.plane_to_nexus, (h.x, sp.x),
            data["hit", "nexus", "sp"].edge_index)

        # message-passing from nexus to interaction
        evt.x = self.checkpoint(
            self.nexus_to_interaction, (sp.x, evt.x),
            data["sp", "in", "evt"].edge_index)

        # message-passing from interaction to nexus
        sp.x = self.checkpoint(
            self.interaction_to_nexus, (evt.x, sp.x),
            data["sp", "in", "evt"].edge_index[(1,0), :])

        # message-passing from nexus to hits
        h.x = self.checkpoint(
            self.nexus_to_plane, (sp.x, h.x),
            data["hit", "nexus", "sp"].edge_index[(1,0), :])

        if not hasattr(h, "of") or not hasattr(h, "ox"):
            raise RuntimeError(
                "NuGraphCore expected data['hit'].of and .ox to be set by Encoder."
            )

        h.of = self.checkpoint(
            self.beta_net, torch.cat((h.of, h.x), dim=1))
        h.ox = self.checkpoint(
            self.coord_net, torch.cat((h.ox, h.x), dim=1))