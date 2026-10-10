"""NuGraph3 semantic feedback from the spacepoint stage to hits"""
import torch
from torch import nn

from .types import T, Data

E_NEXUS = ("hit", "nexus", "sp")
E_SP = ("sp", "sp3d", "sp")


def weighted_mean(x: T, w: T, src: T, dst: T, n: int) -> tuple[T, T]:
    """
    Mean of source node features at each target node, weighted per edge

    Args:
        x: Source node feature tensor
        w: Edge weight tensor
        src: Source node of each edge
        dst: Target node of each edge
        n: Number of target nodes
    """
    num = torch.zeros(n, x.size(1), device=x.device, dtype=x.dtype).index_add_(0, dst, w[:, None] * x[src])
    den = torch.zeros(n, device=x.device, dtype=x.dtype).index_add_(0, dst, w)
    return num / den.clamp(min=1e-6)[:, None], den


class SemanticFeedback(nn.Module):
    """
    Semantic feedback from the spacepoint stage to hits

    Runs after the nexus decoder. Each spacepoint's embedding and its hits'
    mean semantic class probabilities are smoothed over the spacepoint graph,
    weighted by the edge scores, and passed back to each hit as the mean over
    its spacepoints, weighted by how likely each is to be real. A residual
    update, initialised to zero, adds this to the hit features, so that the
    semantic decoder can run again with the spacepoint stage's view of each
    hit's particle. Each hit keeps its own semantic loss, so it can disagree
    with its neighbours. Edge and ghost scores enter only as weights, without
    gradient, so the semantic loss does not train the nexus decoder through
    them.

    Args:
        hit_features: Number of hit node features
        nexus_features: Number of nexus node features
        semantic_classes: Number of semantic classes
        rounds: Number of smoothing rounds over the spacepoint graph
    """
    def __init__(self, hit_features: int, nexus_features: int,
                 semantic_classes: int, rounds: int = 2):
        super().__init__()
        self.rounds = rounds
        features = nexus_features + semantic_classes
        self.sp_net = nn.Sequential(
            nn.Linear(2 * features, features),
            nn.Tanh())
        self.hit_net = nn.Sequential(
            nn.Linear(hit_features + features + 1, hit_features),
            nn.Tanh(),
            nn.Linear(hit_features, hit_features))
        nn.init.zeros_(self.hit_net[-1].weight)
        nn.init.zeros_(self.hit_net[-1].bias)

    def forward(self, data: Data) -> None:
        """
        SemanticFeedback forward pass, updating the hit features in place

        Args:
            data: Graph data object, after the semantic and nexus decoders
        """
        sp, h = data["sp"], data["hit"]
        hit, s = data[E_NEXUS].edge_index
        i, j = data[E_SP].edge_index

        # spacepoint features: embedding and its hits' mean class probabilities
        prob, _ = weighted_mean(h.x_semantic.detach(), torch.ones_like(hit, dtype=sp.x.dtype),
                                hit, s, sp.num_nodes)
        x = torch.cat((sp.x, prob), dim=1)

        # smooth over the spacepoint graph in both directions, weighted by edge score
        score = data[E_SP].x.detach()
        src, dst, w = torch.cat((i, j)), torch.cat((j, i)), torch.cat((score, score))
        for _ in range(self.rounds):
            m, _ = weighted_mean(x, w, src, dst, sp.num_nodes)
            x = self.sp_net(torch.cat((x, m), dim=1))

        # back to hits, weighted by how likely each spacepoint is to be real
        real = sp.x_filter.detach()
        m, den = weighted_mean(x, real[s], s, hit, h.num_nodes)
        linked = (den > 0).to(m.dtype)[:, None]
        h.x = h.x + self.hit_net(torch.cat((h.x, m, linked), dim=1))
