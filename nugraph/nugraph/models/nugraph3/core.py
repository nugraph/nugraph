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
    the (optionally edge-transformed) source node features in order to form
    messages. These messages are then aggregated into the target nodes using
    softmax aggregation, and then fed into a two-layer MLP to generate
    updated target node features.

    An optional edge latent space (edge_features > 0) persists a learned
    embedding on each edge across message-passing iterations. Two flags
    control which sub-networks are active:

    - identity_msg_net=True: message content is raw x_j (Option 3 behaviour)
    - identity_edge_update_net=True: edge embedding is the attention scalar
      passed through unchanged (approaches Option 1 when edge_features=1)

    With both flags False and edge_features > 0, this is the fully general
    Option 2: edges transform message content and maintain their own state.

    Args:
        source_features: Number of source node input features
        target_features: Number of target node input features
        out_features: Number of target node output features
        edge_features_scale: Scale factor for the persistent edge latent space
            dimension, computed as int(scale * min(source_features, target_features)).
            0.0 disables edge latent state entirely, recovering original behaviour.
        identity_msg_net: If True, skip msg_net and use raw x_j as message content
        identity_edge_update_net: If True, skip edge_update_net and persist the
            raw attention scalar as the edge embedding (requires edge_features_scale > 0;
            has no effect when scale=0 since there is no edge state to update)
    """
    def __init__(self, source_features: int, target_features: int,
                 out_features: int, edge_features_scale: float = 0.0,
                 input_edge_features: int = 0,
                 identity_msg_net: bool = False,
                 identity_edge_update_net: bool = False):
        super().__init__(aggr="softmax")

        # scale=0 always means no edge state; identity_edge_update_net changes the update
        # mechanism (store alpha instead of MLP), not whether the edge state is enabled
        if edge_features_scale == 0.0:
            edge_features = 0
        elif identity_edge_update_net:
            edge_features = 1
        else:
            edge_features = int(edge_features_scale * min(source_features, target_features))

        self.edge_features = edge_features
        self.input_edge_features = input_edge_features
        self.identity_msg_net = identity_msg_net
        self.identity_edge_update_net = identity_edge_update_net

        # input_edge_features are fixed (never updated), concatenated alongside learned edge_attr
        self.edge_net = nn.Sequential(
            nn.Linear(source_features + target_features + edge_features + input_edge_features, 1),
            nn.Sigmoid())

        # transforms source features using edge context before aggregation;
        # skipped when identity_msg_net=True (Option 3 / Option 1 behaviour)
        if not identity_msg_net and (edge_features > 0 or input_edge_features > 0):
            self.msg_net = nn.Sequential(
                nn.Linear(source_features + edge_features + input_edge_features, out_features),
                nn.Mish())
        else:
            self.msg_net = None

        # updates the edge embedding each iteration from node and edge context;
        # skipped when identity_edge_update_net=True, where the attention scalar
        # is stored directly as the edge embedding
        if edge_features > 0 and not identity_edge_update_net:
            self.edge_update_net = nn.Sequential(
                nn.Linear(source_features + target_features + edge_features, edge_features),
                nn.Mish(),
                nn.Linear(edge_features, edge_features),
                nn.Mish())
        else:
            self.edge_update_net = None

        msg_features = out_features if (self.msg_net is not None) else source_features
        self.net = nn.Sequential(
            nn.Linear(msg_features + target_features, out_features),
            nn.Mish(),
            nn.Linear(out_features, out_features),
            nn.Mish())

    def forward(self, x: T, edge_index: T, edge_attr: T = None, edge_geom: T = None) -> T: # pylint: disable=arguments-differ
        """
        NuGraphBlock forward pass

        Args:
            x: Node feature tensor (or tuple of source/target tensors)
            edge_index: Edge index tensor
            edge_attr: Persistent edge embedding tensor of shape (E, edge_features),
                or None when edge_features=0
            edge_geom: Fixed geometric edge features of shape (E, input_edge_features),
                or None when input_edge_features=0
        """
        self._new_edge_attr = None
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_geom=edge_geom)

    def message(self, x_i: T, x_j: T, edge_attr: T, edge_geom: T = None) -> T: # pylint: disable=arguments-differ
        """
        NuGraphBlock message function

        Computes per-edge attention weights and message content. If an edge
        latent space is active, the edge embedding is also updated here and
        stored in self._new_edge_attr for write-back by NuGraphCore.
        Fixed geometric features (edge_geom) influence attention and message
        content but are never updated.

        Args:
            x_i: Target node features on each edge
            x_j: Source node features on each edge
            edge_attr: Current edge embedding (None when edge_features=0)
            edge_geom: Fixed geometric edge features (None when input_edge_features=0)
        """
        attn_input = [x_i, x_j]
        if edge_attr is not None:
            attn_input.append(edge_attr)
        if edge_geom is not None:
            attn_input.append(edge_geom)
        alpha = self.edge_net(torch.cat(attn_input, dim=1))

        # update learned edge embedding: either via learned MLP or by storing alpha directly;
        # edge_geom is intentionally excluded — it is a fixed input, never updated
        if self.edge_features > 0:
            if self.identity_edge_update_net:
                self._new_edge_attr = alpha  # alpha is already (E, 1)
            else:
                self._new_edge_attr = self.edge_update_net(
                    torch.cat((x_i, x_j, edge_attr), dim=1))

        # construct message content: edge-transformed x_j or raw x_j
        if self.msg_net is not None:
            msg_input = [x_j]
            if edge_attr is not None:
                msg_input.append(edge_attr)
            if edge_geom is not None:
                msg_input.append(edge_geom)
            msg = self.msg_net(torch.cat(msg_input, dim=1))
        else:
            msg = x_j

        return alpha * msg

    def update(self, aggr_out: T, x: T) -> T: # pylint: disable=arguments-differ
        """
        NuGraphBlock update function

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

    This is the core NuGraph message-passing loop. An optional edge latent
    space can be enabled per edge type by setting edge_features > 0. The
    identity_msg_net and identity_edge_update_net flags further simplify the
    edge sub-networks (see NuGraphBlock for the full option hierarchy).

    Args:
        hit_features: Number of features in planar embedding
        nexus_features: Number of features in nexus embedding
        interaction_features: Number of features in interaction embedding
        instance_features: Number of features in instance embedding
        edge_features_scale: Scale factor for edge latent space dimension, computed
            per block as int(scale * min(source_features, target_features)).
            0.0 disables edge latent state entirely (default, original behaviour).
        identity_msg_net: Pass raw x_j as message content (no msg_net MLP)
        identity_edge_update_net: Store attention scalar as edge embedding instead of using a
            learned update MLP. Requires edge_features_scale > 0; no-op otherwise.
        use_checkpointing: Whether to use gradient checkpointing
        instance_edge_pass: If True, run a dedicated hit-hit message-passing step after each
            main iteration to update condensation coordinates (h.ox) using the hit-hit edge
            state and geometric features as fixed, read-only context. Operates per-edge so
            the model can selectively attend toward same-particle neighbours. Has no effect
            if neither edge_features_scale > 0 nor input_edge_geom is set.
    """
    def __init__(self,
                 hit_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 instance_features: int,
                 edge_features_scale: float = 0.0,
                 input_edge_geom: bool = False,
                 identity_msg_net: bool = False,
                 identity_edge_update_net: bool = False,
                 use_checkpointing: bool = True,
                 instance_edge_pass: bool = False):
        super().__init__()

        self.use_checkpointing = use_checkpointing

        # internal planar message-passing; geometric input edge features only on hit-hit edges
        self.plane_net = NuGraphBlock(hit_features, hit_features, hit_features,
                                      edge_features_scale=edge_features_scale,
                                      input_edge_features=5 if input_edge_geom else 0,
                                      identity_msg_net=identity_msg_net,
                                      identity_edge_update_net=identity_edge_update_net)

        # message-passing from planar nodes to nexus nodes
        self.plane_to_nexus = NuGraphBlock(hit_features, nexus_features, nexus_features,
                                           edge_features_scale=edge_features_scale,
                                           identity_msg_net=identity_msg_net,
                                           identity_edge_update_net=identity_edge_update_net)

        # message-passing from nexus nodes to interaction nodes
        self.nexus_to_interaction = NuGraphBlock(nexus_features, interaction_features,
                                                 interaction_features,
                                                 edge_features_scale=edge_features_scale,
                                                 identity_msg_net=identity_msg_net,
                                                 identity_edge_update_net=identity_edge_update_net)

        # message-passing from interaction nodes to nexus nodes
        self.interaction_to_nexus = NuGraphBlock(interaction_features, nexus_features,
                                                 nexus_features,
                                                 edge_features_scale=edge_features_scale,
                                                 identity_msg_net=identity_msg_net,
                                                 identity_edge_update_net=identity_edge_update_net)

        # message-passing from nexus nodes to planar nodes
        self.nexus_to_plane = NuGraphBlock(nexus_features, hit_features, hit_features,
                                           edge_features_scale=edge_features_scale,
                                           identity_msg_net=identity_msg_net,
                                           identity_edge_update_net=identity_edge_update_net)

        # dedicated instance post-pass: updates h.ox using hit-hit edge context as fixed input;
        # edge_attr (learned state) and edge_geom (geometric features) are concatenated and
        # passed as read-only — this block never writes back to the edge store
        inst_edge_ctx = self.plane_net.edge_features + (5 if input_edge_geom else 0)
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

    def checkpoint(self, net: nn.Module, x: T, edge_store=None, *, reverse: bool = False) -> TD:
        """
        Checkpoint module, if enabled, and write back updated edge embedding.

        Args:
            net: Network module
            x: Node feature tensor (or tuple of source/target tensors)
            edge_store: Edge data store supplying edge_index and edge_attr,
                or None for plain nn.Sequential modules (beta_net, coord_net)
            reverse: If True, reverse the edge direction (for down-passes)
        """
        if edge_store is not None:
            edge_index = edge_store.edge_index[(1, 0), :] if reverse else edge_store.edge_index
            # cross-type edges keep separate fwd/bwd embeddings (encoder initialises edge_attr_fwd);
            # same-type edges (hit-hit) use a single symmetric edge_attr
            if edge_store.get("edge_attr_fwd", None) is not None:
                attr_key = "edge_attr_bwd" if reverse else "edge_attr_fwd"
            else:
                attr_key = "edge_attr"
            edge_attr = edge_store.get(attr_key, None)
            edge_geom = edge_store.get("edge_geom", None)
            if self.use_checkpointing and self.training:
                result = checkpoint(net, x, edge_index, edge_attr, edge_geom, use_reentrant=False)
            else:
                result = net(x, edge_index, edge_attr, edge_geom)
            if net.edge_features > 0 and net._new_edge_attr is not None:
                setattr(edge_store, attr_key, net._new_edge_attr)
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
            data["hit", "delaunay-planar", "hit"])

        # message-passing from hits to nexus
        sp.x = self.checkpoint(
            self.plane_to_nexus, (h.x, sp.x),
            data["hit", "nexus", "sp"])

        # message-passing from nexus to interaction
        evt.x = self.checkpoint(
            self.nexus_to_interaction, (sp.x, evt.x),
            data["sp", "in", "evt"])

        # message-passing from interaction to nexus
        sp.x = self.checkpoint(
            self.interaction_to_nexus, (evt.x, sp.x),
            data["sp", "in", "evt"], reverse=True)

        # message-passing from nexus to hits
        h.x = self.checkpoint(
            self.nexus_to_plane, (sp.x, h.x),
            data["hit", "nexus", "sp"], reverse=True)

        if not hasattr(h, "of") or not hasattr(h, "ox"):
            raise RuntimeError(
                "NuGraphCore expected data['hit'].of and .ox to be set by Encoder."
            )

        h.of = self.checkpoint(
            self.beta_net, torch.cat((h.of, h.x), dim=1))
        h.ox = self.checkpoint(
            self.coord_net, torch.cat((h.ox, h.x), dim=1))

        if self.instance_net is not None:
            pp = data["hit", "delaunay-planar", "hit"]
            # combine learned edge state and geometric features as a single fixed context;
            # this block reads them but never updates the edge store
            ctx = [t for t in (pp.get("edge_attr", None), pp.get("edge_geom", None))
                   if t is not None]
            inst_edge_ctx = torch.cat(ctx, dim=1) if ctx else None
            inst_in = torch.cat([h.ox, h.x], dim=1)
            if self.use_checkpointing and self.training:
                h.ox = checkpoint(self.instance_net, inst_in, pp.edge_index,
                                  None, inst_edge_ctx, use_reentrant=False)
            else:
                h.ox = self.instance_net(inst_in, pp.edge_index, None, inst_edge_ctx)