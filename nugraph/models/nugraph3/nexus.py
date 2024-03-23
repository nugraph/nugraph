from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from torch_geometric.nn import MessagePassing, SimpleConv, AttentionalAggregation

T = torch.Tensor
TD = dict[str, T]

class NexusDown(MessagePassing):
    def __init__(self,
                 planar_features: int,
                 nexus_features: int,
                 aggr: str = 'mean'):
        super().__init__(node_dim=0, aggr=aggr, flow='target_to_source')

        feats = planar_features + nexus_features

        self.edge_net = nn.Sequential(
            nn.Linear(feats, nexus_features),
            nn.Sigmoid(),
        )

    def forward(self, x: T, edge_index: T, n: T) -> T:
        return self.propagate(x=x, n=n, edge_index=edge_index)

    def message(self, x_i: T, n_j: T) -> T:
        return self.edge_net(torch.cat((x_i, n_j), dim=-1).detach()) * n_j

    def update(self, aggr_out: T) -> T:
        return aggr_out

class NexusNet(nn.Module):
    '''Module to project to nexus space and mix detector planes'''
    def __init__(self,
                 planar_features: int,
                 nexus_features: int,
                 planes: list[str],
                 aggr: str = 'mean',
                 checkpoint: bool = True):
        super().__init__()

        self.checkpoint = checkpoint

        self.nexus_up = SimpleConv(node_dim=0)

        self.nexus_net = nn.Sequential(
            nn.Linear(len(planes)*planar_features, nexus_features),
            nn.Tanh(),
            nn.Linear(nexus_features, nexus_features),
            nn.Tanh(),
        )

        self.nexus_down = nn.ModuleDict()
        for p in planes:
            self.nexus_down[p] = NexusDown(planar_features,
                                           nexus_features,
                                           aggr)

    def ckpt(self, fn: Callable, *args) -> Any:
        if self.checkpoint and self.training:
            return checkpoint(fn, *args)
        else:
            return fn(*args)

    def forward(self, x: TD, edge_index: TD, nexus: T) -> TD:

        # project up to nexus space
        n = [None] * len(self.nexus_down)
        for i, p in enumerate(self.nexus_down):
            n[i] = self.nexus_up(x=(x[p], nexus), edge_index=edge_index[p])

        # convolve in nexus space
        n = self.ckpt(self.nexus_net, torch.cat(n, dim=-1))

        # project back down to planes
        ret = dict()
        for p in self.nexus_down:
            ret[p] = self.ckpt(self.nexus_down[p], x[p], edge_index[p], n)
        return ret
