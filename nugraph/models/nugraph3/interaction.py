import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# from torch_geometric.nn import MessagePassing, SimpleConv, MLP
from torch_geometric.nn.aggr import MeanAggregation

T = torch.Tensor
TD = dict[str, T]

# class InteractionUp(MessagePassing):
#     def __init__(self,
#                  planar_features: int,
#                  planes: list[str],
#                  aggr: str = 'mean') -> None:
#         super().__init__(node_dim=0, aggr=aggr)

#         self.edge_net = nn.Sequential(
#             nn.Linear(planar_features, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x: T, edge_index: T) -> T:
#         return self.propagate(x=x, edge_index=edge_index)

#     def message(self, x_j: T) -> T:
#         return self.edge_net(x_j).detach() * x_j

#     def update(self, aggr_out: T) -> T:
#         print(aggr_out.shape)
#         return aggr_out

class InteractionNet(nn.Module):
    def __init__(self,
                 planar_features: int,
                 interaction_features: int,
                 planes: list[str]) -> None:
        super().__init__()

        self.att = nn.ModuleDict()
        for p in planes:
            self.att[p] = nn.Sequential(
                nn.Linear(planar_features, 1),
                nn.Sigmoid(),
            )
        self.aggr = MeanAggregation()
        
        self.net = nn.Linear(len(planes) * planar_features, interaction_features)

    def forward(self, x: TD, index: TD) -> T:
        e = [ self.aggr(net(x[p])*x[p], index=index[p]) for p, net in self.att.items() ]
        e = torch.cat(e, dim=1)
        return self.net(e)
