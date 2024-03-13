from torch import Tensor
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,
                 in_features: int,
                 node_features: int,
                 planes: list[str]):
        super().__init__()

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Sequential(
                nn.Linear(in_features, node_features),
                nn.Tanh(),
            )

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        return { p: net(x[p]) for p, net in self.net.items() }