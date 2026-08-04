"""NuGraph2 nexus module"""
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
        return self.edge_net(torch.cat((x_i, n_j), dim=-1)) * n_j

    def update(self, aggr_out: T, x: T) -> T: # pylint: disable=arguments-differ
        return self.node_net(torch.cat((x, aggr_out), dim=-1))

class NexusNet(torch.nn.Module):
    """
    Module to project to nexus space and mix detector planes

    Args:
        planar_features: Number of planar features
        nexus_features: Number of nexus features
        num_classes: Number of semantic classes
        planes: Tuple of plane names
        aggr: Message aggregation method
        checkpoint: Whether to use checkpointing
    """
    def __init__(self, # pylint: disable=too-many-arguments,too-many-positional-arguments
                 planar_features: int,
                 nexus_features: int,
                 num_classes: int,
                 planes: tuple[str],
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

    def forward(self, x: dict[str, T], edge_index: dict[str, T], nexus: T) -> None:
        """
        NuGraph2 nexus module forward pass

        Args:
            x: Planar embedding tensor dictionary
            edge_index: Edge indices mapping planar nodes to nexus nodes
            nexus: Nexus embedding tensor
        """

        # project up to nexus space
        n: list[T] = [torch.empty(0)] * len(self.nexus_down)
        for i, p in enumerate(self.nexus_down):
            n[i] = self.nexus_up(x=(x[p], nexus), edge_index=edge_index[p])

        # convolve in nexus space; torch.jit.is_scripting() guard makes checkpoint branch dead code at compile time
        x_cat = torch.cat(n, dim=-1)
        if not torch.jit.is_scripting() and self.checkpoint and self.training:
            n_cat = torch.utils.checkpoint.checkpoint(self.nexus_net, x_cat, use_reentrant=False)
        else:
            n_cat = self.nexus_net(x_cat)

        # project back down to planes
        for p, net in self.nexus_down.items():
            if not torch.jit.is_scripting() and self.checkpoint and self.training:
                x[p] = torch.utils.checkpoint.checkpoint(net, x[p], edge_index[p], n_cat, use_reentrant=False)
            else:
                x[p] = net.forward(x[p], edge_index[p], n_cat)
