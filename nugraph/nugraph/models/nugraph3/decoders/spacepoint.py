"""NuGraph3 spacepoint decoder"""
from typing import Any
import torch
from torch import nn
from torch_geometric.data import Batch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from ....util import LogCoshLoss
from ..types import Data

class SpacepointDecoder(LightningModule):
    """
    NuGraph3 spacepoint decoder module

    Convolve planar node embedding down to coordinates in 3D Euclidean space

    Args:
        hit_features: Number of planar hit node features
    """
    def __init__(self, hit_features: int):
        super().__init__()

        # loss function
        self.loss = LogCoshLoss()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(5.))

        # network
        self.net = nn.Linear(hit_features, 3)

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]: # pylint: disable=arguments-differ
        """
        NuGraph3 spacepoint decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        # add predicted coordinates to output graph
        data["hit"].c_pred = self.net(data["hit"].x)
        if isinstance(data, Batch):
            # pylint: disable=protected-access
            data._slice_dict["hit"]["c_pred"] = data["hit"].ptr
            inc = torch.zeros(data.num_graphs, device=self.device)
            data._inc_dict["hit"]["c_pred"] = inc

        # calculate loss
        mask = data["hit"].y_semantic != -1
        x = data["hit"].c_pred[mask]
        y = data["hit"].c[mask]
        loss = (-1 * self.temp).exp() * self.loss(x, y) + self.temp



        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"spacepoint/loss-{stage}"] = loss
            xyz = (x-y).abs().mean(dim=0)
            metrics[f"spacepoint/x-resolution-{stage}"] = xyz[0]
            metrics[f"spacepoint/y-resolution-{stage}"] = xyz[1]
            metrics[f"spacepoint/z-resolution-{stage}"] = xyz[2]
            metrics[f"spacepoint/resolution-{stage}"] = xyz.square().sum().sqrt()
        if stage == "train":
            metrics["temperature/spacepoint"] = self.temp

        return loss, metrics

    def on_epoch_end(self, logger: WandbLogger, stage: str,
                     epoch: int) -> None: # pylint: disable=unused-argument
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: Wandb logger object
            stage: Training stage
            epoch: Training epoch index
        """
