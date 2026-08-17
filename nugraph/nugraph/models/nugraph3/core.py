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
    the source and target node features (and optional fixed geometric features),
    and then applies those weights to the source node features in order to form
    messages. These messages are then aggregated into the target nodes using
    softmax aggregation, and then fed into a two-layer MLP to generate updated
    target node features. Fixed geometric features (input_edge_features > 0)
    influence both attention weights and message content but are never updated.

    Args:
        source_features: Number of source node input features
        target_features: Number of target node input features
        out_features: Number of target node output features
        input_edge_features: Number of fixed (non-updated) input edge features
            appended to attention and message inputs. 0 disables.
    """
    def __init__(self, source_features: int, target_features: int,
                 out_features: int, input_edge_features: int = 0):
        super().__init__(aggr="softmax")

        self.input_edge_features = input_edge_features

        # attention: scalar weight per edge from node and edge context
        self.edge_net = nn.Sequential(
            nn.Linear(source_features+target_features+input_edge_features, 1),
            nn.Sigmoid())

        # transforms source features using fixed edge context before aggregation;
        # only created when edge context is available
        if input_edge_features > 0:
            self.msg_net = nn.Sequential(
                nn.Linear(source_features+input_edge_features, out_features),
                nn.Mish())
        else:
            self.msg_net = None

        msg_features = out_features if (self.msg_net is not None) else source_features
        self.net = nn.Sequential(
            nn.Linear(msg_features+target_features, out_features),
            nn.Mish(),
            nn.Linear(out_features, out_features),
            nn.Mish())

    def forward(self, x: T, edge_index: T, edge_geom: T = None) -> T: # pylint: disable=arguments-differ
        """
        NuGraphBlock forward pass

        Args:
            x: Node feature tensor (or tuple of source/target tensors)
            edge_index: Edge index tensor
            edge_geom: Fixed geometric edge features of shape (E, input_edge_features),
                or None when input_edge_features=0
        """
        return self.propagate(edge_index, x=x, edge_geom=edge_geom)

    def message(self, x_i: T, x_j: T, edge_geom: T = None) -> T: # pylint: disable=arguments-differ
        """
        NuGraphBlock message function

        This function constructs messages on graph edges. Features from the
        source and target nodes are concatenated and fed into a linear layer
        to construct attention weights. Messages are then formed on edges by
        weighting the source node features by these attention weights.
        Fixed geometric edge features are optionally used.

        Args:
            x_i: Target node features on each edge
            x_j: Source node features on each edge
            edge_geom: Fixed geometric edge features (None when input_edge_features=0)
        """
        attn_input = [x_i, x_j]
        if edge_geom is not None:
            attn_input.append(edge_geom)
        alpha = self.edge_net(torch.cat(attn_input, dim=1))

        if self.msg_net is not None:
            msg = self.msg_net(torch.cat([x_j, edge_geom], dim=1))
        else:
            msg = x_j

        return alpha * msg

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
        instance_features: Number of features in instance embedding
        input_edge_geom: If True, inject 5 fixed geometric features on hit-hit edges
        instance_edge_pass: If True, run a dedicated hit-hit message-passing step after each
            main iteration to update condensation coordinates (h.ox) using geometric edge
            features as fixed, read-only context. Has no effect if input_edge_geom is False.
        use_checkpointing: Whether to use gradient checkpointing
    """
    def __init__(self,
                 hit_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 instance_features: int,
                 input_edge_geom: bool = False,
                 instance_edge_pass: bool = False,
                 use_checkpointing: bool = True):
        super().__init__()

        self.use_checkpointing = use_checkpointing

        # internal planar message-passing; geometric input edge features only on hit-hit edges
        self.plane_net = NuGraphBlock(hit_features, hit_features, hit_features,
                                      input_edge_features=5 if input_edge_geom else 0)

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

        # dedicated instance post-pass: updates h.ox using geometric edge features as fixed input;
        # this block reads edge_geom but never writes back to the edge store
        inst_edge_ctx = 5 if input_edge_geom else 0
        if instance_edge_pass and inst_edge_ctx > 0:
            self.instance_net = NuGraphBlock(
                hit_features + instance_features,
                hit_features + instance_features,
                instance_features,
                input_edge_features=inst_edge_ctx)
        else:
            self.instance_net = None

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

    def checkpoint(self, net: nn.Module, x: T, edge_index: T = None, edge_geom: T = None) -> T:
        """
        Checkpoint module, if enabled.

        Args:
            net: Network module
            x: Node feature tensor (or tuple of source/target tensors)
            edge_index: Edge index tensor, or None for plain nn.Sequential modules
            edge_geom: Fixed geometric edge features, or None
        """
        if edge_index is not None:
            if self.use_checkpointing and self.training:
                result = checkpoint(net, x, edge_index, edge_geom, use_reentrant=False)
            else:
                result = net(x, edge_index, edge_geom)
        else:
            if self.use_checkpointing and self.training:
                result = checkpoint(net, x, use_reentrant=False)
            else:
                result = net(x)
        return result

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
            data["hit", "delaunay-planar", "hit"].edge_index,
            data["hit", "delaunay-planar", "hit"].get("edge_geom", None))

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

        if self.instance_net is not None:
            h.ox = self.checkpoint(
                self.instance_net, torch.cat([h.ox, h.x], dim=1),
                data["hit", "delaunay-planar", "hit"].edge_index,
                data["hit", "delaunay-planar", "hit"].get("edge_geom", None))
