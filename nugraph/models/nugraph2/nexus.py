from typing import Any, Callable

from torch import Tensor, cat
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from torch_geometric.nn import MessagePassing, SimpleConv

from .linear import ClassLinear

class NexusDown(MessagePassing):
    def __init__(self,
                 planar_features: int,
                 nexus_features: int,
                 num_classes: int,
                 aggr: str = 'mean'):
        super().__init__(node_dim=0, aggr=aggr, flow='target_to_source')

        self.edge_net = nn.Sequential(
            ClassLinear(planar_features+nexus_features,
                        1,
                        num_classes),
            nn.Softmax(dim=1))

        self.node_net = nn.Sequential(
            ClassLinear(planar_features+nexus_features,
                        planar_features,
                        num_classes),
            nn.Tanh(),
            ClassLinear(planar_features,
                        planar_features,
                        num_classes),
            nn.Tanh())

    def forward(self, x: Tensor, edge_index: Tensor, n: Tensor) -> Tensor:
        return self.propagate(edge_index=edge_index, x=x, n=n)

    def message(self, x_i: Tensor, n_j: Tensor) -> Tensor:
        return self.edge_net(cat((x_i, n_j), dim=-1).detach()) * n_j

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return self.node_net(cat((x, aggr_out), dim=-1))

class NexusNet(nn.Module):
    '''Module to project to nexus space and mix detector planes'''
    def __init__(self,
                 planar_features: int,
                 nexus_features: int,
                 num_classes: int,
                 planes: list[str],
                 aggr: str = 'mean',
                 checkpoint: bool = True):
        super().__init__()

        self.checkpoint = checkpoint

        self.nexus_up = SimpleConv(node_dim=0)

        self.nexus_net = nn.Sequential(
            ClassLinear(len(planes)*planar_features,
                        nexus_features,
                        num_classes),
            nn.Tanh(),
            ClassLinear(nexus_features,
                        nexus_features,
                        num_classes),
            nn.Tanh())

        self.nexus_down = nn.ModuleDict()
        for p in planes:
            self.nexus_down[p] = NexusDown(planar_features,
                                           nexus_features,
                                           num_classes,
                                           aggr)

    def ckpt(self, fn: Callable, *args) -> Any:
        if self.checkpoint and self.training:
            return checkpoint(fn, *args)
        else:
            return fn(*args)

    def forward(self, x: dict[str, Tensor], edge_index: dict[str, Tensor], nexus: Tensor) -> None:

        # project up to nexus space
        n = [None] * len(self.nexus_down)
        for i, p in enumerate(self.nexus_down):
            n[i] = self.nexus_up(x=(x[p], nexus), edge_index=edge_index[p])

        # convolve in nexus space
        n = self.ckpt(self.nexus_net, cat(n, dim=-1))

        # project back down to planes
        for p in self.nexus_down:
            x[p] = self.ckpt(self.nexus_down[p], x[p], edge_index[p], n)
