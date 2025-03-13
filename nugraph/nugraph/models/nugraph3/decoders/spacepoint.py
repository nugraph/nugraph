"""NuGraph3 spacepoint decoder"""
from typing import Any
import torch
from torch_geometric.data import Batch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pynuml.data import NuGraphData
from ....util import LogCoshLoss

class SpacepointDecoder(LightningModule):
    """
    NuGraph3 spacepoint decoder module

    Convolve planar node embedding down to coordinates in 3D Euclidean space

    Args:
        hit_features: Number of planar hit node features
        num_planes: Number of detector planes
    """
    def __init__(self, hit_features: int, num_planes: int):
        super().__init__()

        # loss function
        self.loss = LogCoshLoss()

        # temperature parameter
        self.temp = torch.nn.Parameter(torch.tensor(6.))

        # network
        self.net = torch.nn.ModuleList(
            [torch.nn.Linear(hit_features, 3) for _ in range(num_planes)])

    def forward(self, data: NuGraphData, stage: str = None) -> dict[str, Any]: # pylint: disable=arguments-differ
        """
        NuGraph3 spacepoint decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        # add predicted coordinates to output graph
        h = data["hit"]
        h.x_position = torch.empty((h.num_nodes, 3), device=self.device, dtype=torch.float)
        for i, net in enumerate(self.net):
            mask = h.plane == i
            h.x_position[mask] = net(h.x[mask])
        if isinstance(data, Batch):
            # pylint: disable=protected-access
            data._slice_dict["hit"]["x_position"] = h.ptr
            inc = torch.zeros(data.num_graphs, device=self.device)
            data._inc_dict["hit"]["x_position"] = inc

        # calculate loss
        mask = h.y_semantic != -1
        x = h.x_position[mask]
        y = h.y_position[mask]
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
