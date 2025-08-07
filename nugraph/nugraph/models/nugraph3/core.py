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
                 instance_features: int,
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

        # message-passing from nexus nodes to planar nodes
        self.nexus_to_plane = NuGraphBlock(nexus_features, hit_features,
                                           hit_features)
        
        self.beta_net = NuGraphBlock(1, 1, 1)
        self.coord_net = NuGraphBlock(instance_features, instance_features, instance_features)

        self.sem_to_beta = NuGraphBlock(hit_features, 1, 1)
        self.sem_to_coord = NuGraphBlock(hit_features, instance_features, instance_features)

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
        x = data["hit"].x       # semantic
        of = data["hit"].of     # beta
        ox = data["hit"].ox     # coord
        edge_planar = data["hit", "delaunay-planar", "hit"].edge_index
        edge_nexus = data["hit", "nexus", "sp"].edge_index

        # Planar semantic message passing
        x = self.checkpoint(self.plane_net, x, edge_planar)

        # x to sp, then sp to x
        data["sp"].x = self.checkpoint(self.plane_to_nexus, (x, data["sp"].x), edge_nexus)
        x = self.checkpoint(self.nexus_to_plane, (data["sp"].x, x), edge_nexus[(1, 0), :])
        
        # OC message passing (beta and coord)
        of = self.checkpoint(self.beta_net, of, edge_planar)
        ox = self.checkpoint(self.coord_net, ox, edge_planar)

        # Cross: semantic to OC
        of = self.checkpoint(self.sem_to_beta, (x, of), edge_planar)
        ox = self.checkpoint(self.sem_to_coord, (x, ox), edge_planar)

        # Update data object ~ Bug
        data["hit"].x = x
        data["hit"].of = torch.sigmoid(of)
        data["hit"].ox = ox