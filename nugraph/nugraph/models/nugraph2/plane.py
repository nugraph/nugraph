"""NuGraph2 planar module"""
from typing import Any, Callable
import torch
from torch_geometric.nn import MessagePassing
from .linear import ClassLinear

T = torch.Tensor

class MessagePassing2D(MessagePassing): # pylint: disable=abstract-method
    """
    Message-passing module for NuGraph2 planar step

    Args:
        in_features: Number of input features
        planar_features: Number of planar features
        num_classes: Number of semantic classes
        aggr: Message aggregation method
    """
    propagate_type = {'x': T}

    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 num_classes: int,
                 aggr: str = 'add'):
        super().__init__(node_dim=0, aggr=aggr)

        self.edge_net = torch.nn.Sequential(
            ClassLinear(2 * (in_features + planar_features),
                        1,
                        num_classes),
            torch.nn.Softmax(dim=1))

        self.node_net = torch.nn.Sequential(
            ClassLinear(2 * (in_features + planar_features),
                        planar_features,
                        num_classes),
            torch.nn.Tanh(),
            ClassLinear(planar_features,
                        planar_features,
                        num_classes),
            torch.nn.Tanh())

    def forward(self, x: T, edge_index: T): # pylint: disable=arguments-differ
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: T, x_j: T): # pylint: disable=arguments-differ
        return self.edge_net(torch.cat((x_i, x_j), dim=-1).detach()) * x_j

    def update(self, aggr_out: T, x: T): # pylint: disable=arguments-differ
        return self.node_net(torch.cat((x, aggr_out), dim=-1))

class PlaneNet(torch.nn.Module):
    """
    Module to pass messages within each detector plane

    Args:
        in_features: Number of input features
        planar_features: Number of planar features
        num_classes: Number of semantic classes
        planes: Tuple of plane names
        aggr: Message aggregation method
        checkpoint: Whether to use checkpointing
    """
    def __init__(self, # pylint: disable=too-many-arguments,too-many-positional-arguments
                 in_features: int,
                 planar_features: int,
                 num_classes: int,
                 planes: tuple[str],
                 aggr: str = 'add',
                 checkpoint: bool = True):
        super().__init__()

        self.checkpoint = checkpoint

        self.net = torch.nn.ModuleDict()
        for p in planes:
            self.net[p] = MessagePassing2D(in_features,
                                           planar_features,
                                           num_classes,
                                           aggr)

    def ckpt(self, fn: Callable, *args) -> Any:
        """
        NuGraph2 planar module checkpointing function

        Args:
            fn: Module to checkpoint
            args: Module arguments
        """
        if self.checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)

        return fn(*args)

    def forward(self, x: dict[str, T], edge_index: dict[str, T]) -> None:
        """
        NuGraph2 planar module forward pass

        Args:
            x: Planar embedding tensor dictionary
            edge_index: Edge indices within each plane
        """
        for p in self.net:
            x[p] = self.ckpt(self.net[p], x[p], edge_index[p])
