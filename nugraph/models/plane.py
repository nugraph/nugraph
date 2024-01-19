from typing import Any, Callable

from torch import Tensor, cat
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from torch_geometric.nn import MessagePassing

class MessagePassing2D(MessagePassing):

    propagate_type = { 'x': Tensor }

    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 aggr: str = 'add'):
        super().__init__(node_dim=0, aggr=aggr)

        feats = 2 * (in_features + planar_features)

        self.edge_net = nn.Sequential(
            nn.Linear(feats, 1),
            nn.Sigmoid(),
        )

        self.node_net = nn.Sequential(
            nn.Linear(feats, planar_features),
            nn.Tanh(),
            nn.Linear(planar_features, planar_features),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, edge_index: Tensor):
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor):
        return self.edge_net(cat((x_i, x_j), dim=-1).detach()) * x_j

    def update(self, aggr_out: Tensor, x: Tensor):
        return self.node_net(cat((x, aggr_out), dim=-1))

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
        else:
            return fn(*args)

    def forward(self, x: dict[str, Tensor], edge_index: dict[str, Tensor]) -> None:
        for p in self.net:
            x[p] = self.ckpt(self.net[p], x[p], edge_index[p])
