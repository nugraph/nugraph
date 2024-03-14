from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from torch_geometric.nn import MessagePassing

T = torch.Tensor
TD = dict[str, T]

class MessagePassing2D(MessagePassing):

    propagate_type = { 'x': T }

    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 aggr: str = 'add'):
        super().__init__(node_dim=0, aggr=aggr)

        self.edge_net = nn.Sequential(
            nn.Linear(2 * planar_features, 1),
            nn.Sigmoid(),
        )

        self.node_net = nn.Sequential(
            nn.Linear(2 * planar_features, planar_features),
            nn.Tanh(),
            nn.Linear(planar_features, planar_features),
            nn.Tanh(),
        )

    def forward(self, x: T, edge_index: T):
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: T, x_j: T):
        return self.edge_net(torch.cat((x_i, x_j), dim=-1).detach()) * x_j

    def update(self, aggr_out: T, x: T):
        return self.node_net(torch.cat((x, aggr_out), dim=-1))

class PlaneNet(nn.Module):
    '''Module to convolve within each detector plane'''
    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 planes: list[str],
                 aggr: str = 'add',
                 checkpoint: bool = True):
        super().__init__()

        self.checkpoint = checkpoint

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = MessagePassing2D(in_features,
                                           planar_features,
                                           aggr)

    def ckpt(self, fn: Callable, *args) -> Any:
        if self.checkpoint and self.training:
            return checkpoint(fn, *args)
        return fn(*args)

    def forward(self, x: TD, edge_index: TD) -> TD:
        ret = dict()
        for p, net in self.net.items():
            ret[p] = self.ckpt(net, x[p], edge_index[p])
        return ret
