"""NuGraph3 vertex decoder"""
from typing import Any
import torch
from torch import nn
from torch_geometric.data import Batch
from pytorch_lightning.loggers import TensorBoardLogger
from ....util import LogCoshLoss
from ..types import Data

class VertexDecoder(nn.Module):
    """
    NuGraph3 vertex decoder module

    Convolve interaction node embedding down to a set of 3D coordinates
    predicting the location of the neutrino interaction vertex.

    Args:
        interaction_features: Number of interaction node features
    """
    def __init__(self, interaction_features: int):
        super().__init__()

        # loss function
        self.loss = LogCoshLoss()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(5.))

        # network
        self.net = nn.Linear(interaction_features, 3)

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 vertex decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        # run network and calculate loss
        x = self.net(data["evt"].x)
        y = data["evt"].y_vtx
        w = (-1 * self.temp).exp()
        loss = w * self.loss(x, y) + self.temp

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"loss_vertex/{stage}"] = loss
            xyz = (x-y).abs().mean(dim=0)
            metrics[f"vertex-resolution-x/{stage}"] = xyz[0]
            metrics[f"vertex-resolution-y/{stage}"] = xyz[1]
            metrics[f"vertex-resolution-z/{stage}"] = xyz[2]
            metrics[f"vertex-resolution/{stage}"] = xyz.square().sum().sqrt()

        # add inference output to graph object
        data["evt"].v = x
        if isinstance(data, Batch):
            data._slice_dict["evt"]["v"] = data["evt"].ptr
            inc = torch.zeros(data.num_graphs, device=data["evt"].x.device)
            data._inc_dict["evt"]["v"] = inc

        return loss, metrics

    def on_epoch_end(self,
                     logger: TensorBoardLogger,
                     stage: str,
                     epoch: int) -> None:
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: Tensorboard logger object
            stage: Training stage
            epoch: Training epoch index
        """
