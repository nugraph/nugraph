"""NuGraph2 nexus module"""
from typing import Any, Callable
import torch
from torch_geometric.nn import MessagePassing, SimpleConv
from .linear import ClassLinear

T = torch.Tensor

class NexusDown(MessagePassing): # pylint: disable=abstract-method
    """
    Message-passing module for NuGraph2 nexus downward step

    Args:
        planar_features: Number of planar features
        nexus_featues: Number of nexus features
        num_classes: Number of semantic classes
        aggr: Message aggregation method
    """
    def __init__(self,
                 planar_features: int,
                 nexus_features: int,
                 num_classes: int,
                 aggr: str = 'mean'):
        super().__init__(node_dim=0, aggr=aggr, flow='target_to_source')

        self.edge_net = torch.nn.Sequential(
            ClassLinear(planar_features+nexus_features,
                        1,
                        num_classes),
            torch.nn.Softmax(dim=1))

        self.node_net = torch.nn.Sequential(
            ClassLinear(planar_features+nexus_features,
                        planar_features,
                        num_classes),
            torch.nn.Tanh(),
            ClassLinear(planar_features,
                        planar_features,
                        num_classes),
            torch.nn.Tanh())

    def forward(self, x: T, edge_index: T, n: T) -> T: # pylint: disable=arguments-differ
        return self.propagate(edge_index=edge_index, x=x, n=n)

    def message(self, x_i: T, n_j: T) -> T: # pylint: disable=arguments-differ
        return self.edge_net(torch.cat((x_i, n_j), dim=-1).detach()) * n_j

    def update(self, aggr_out: T, x: T) -> T: # pylint: disable=arguments-differ
        return self.node_net(torch.cat((x, aggr_out), dim=-1))

class NexusNet(torch.nn.Module):
    '''Module to project to nexus space and mix detector planes'''
    def __init__(self, # pylint: disable=too-many-arguments,too-many-positional-arguments
                 planar_features: int,
                 nexus_features: int,
                 num_classes: int,
                 planes: list[str],
                 aggr: str = 'mean',
                 checkpoint: bool = True):
        super().__init__()

        self.checkpoint = checkpoint

        self.nexus_up = SimpleConv(node_dim=0)

        self.nexus_net = torch.nn.Sequential(
            ClassLinear(len(planes)*planar_features,
                        nexus_features,
                        num_classes),
            torch.nn.Tanh(),
            ClassLinear(nexus_features,
                        nexus_features,
                        num_classes),
            torch.nn.Tanh())

        self.nexus_down = torch.nn.ModuleDict()
        for p in planes:
            self.nexus_down[p] = NexusDown(planar_features,
                                           nexus_features,
                                           num_classes,
                                           aggr)

    def ckpt(self, fn: Callable, *args) -> Any:
        """
        NuGraph2 nexus module checkpointing function

        Args:
            fn: Module to checkpoint
            args: Module arguments
        """
        if self.checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)

        return fn(*args)

    def forward(self, x: dict[str, T], edge_index: dict[str, T], nexus: T) -> None:
        """
        NuGraph2 nexus module forward pass

        Args:
            x: Planar embedding tensor dictionary
            edge_index: Edge indices mapping planar nodes to nexus nodes
            nexus: Nexus embedding tensor
        """

        # project up to nexus space
        n = [None] * len(self.nexus_down)
        for i, p in enumerate(self.nexus_down):
            n[i] = self.nexus_up(x=(x[p], nexus), edge_index=edge_index[p])

        # convolve in nexus space
        n = self.ckpt(self.nexus_net, torch.cat(n, dim=-1))

        # project back down to planes
        for p in self.nexus_down:
            x[p] = self.ckpt(self.nexus_down[p], x[p], edge_index[p], n)
