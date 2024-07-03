"""NuGraph3 filter decoder"""
from typing import Any
import torch
from torch import nn
import torchmetrics as tm
from torch_geometric.data import Batch
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import seaborn as sn
from ..types import Data

class FilterDecoder(nn.Module):
    """
    NuGraph3 filter decoder module

    Convolve planar node embedding down to a single node score to identify and
    filter out graph nodes that are not part of the primary physics
    interaction.

    Args:
        node_features: Number of planar node features
        planes: List of detector planes
    """
    def __init__(self,
                 node_features: int,
                 planes: list[str]):
        super().__init__()

        # loss function
        self.loss = nn.BCELoss()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.))

        # metrics
        metric_args = {"task": "binary"}
        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.cm_recall = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.cm_precision = tm.ConfusionMatrix(normalize="pred", **metric_args)

        # network
        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Sequential(
                nn.Linear(node_features, 1),
                nn.Sigmoid(),
            )

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 filter decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        # run network and add output to graph object
        for p, net in self.net.items():
            data[p].x_filter = net(data[p].x).squeeze(dim=-1)
            if isinstance(data, Batch):
                data._slice_dict[p]["x_filter"] = data[p].ptr
                inc = torch.zeros(data.num_graphs, device=data[p].x.device)
                data._inc_dict[p]["x_filter"] = inc

        # calculate loss
        x = torch.cat([data[p].x_filter for p in self.net], dim=0)
        y = torch.cat([data[p].y_semantic!=-1 for p in self.net], dim=0).float()
        w = 2 * (-1 * self.temp).exp()
        loss = w * self.loss(x, y) + self.temp

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"loss_filter/{stage}"] = loss
            metrics[f"recall_filter/{stage}"] = self.recall(x, y)
            metrics[f"precision_filter/{stage}"] = self.precision(x, y)
        if stage == "train":
            metrics["temperature/filter"] = self.temp
        if stage in ["val", "test"]:
            self.cm_recall.update(x, y)
            self.cm_precision.update(x, y)

        return loss, metrics

    def draw_confusion_matrix(self, cm: tm.ConfusionMatrix) -> plt.Figure:
        """
        Draw confusion matrix

        Args:
            cm: Confusion matrix object
        """
        confusion = cm.compute().cpu()
        fig = plt.figure(figsize=[8,6])
        sn.heatmap(confusion,
                   xticklabels=("noise", "signal"),
                   yticklabels=("noise", "signal"),
                   vmin=0, vmax=1,
                   annot=True)
        plt.ylim(0, 2)
        plt.xlabel("Assigned label")
        plt.ylabel("True label")
        return fig

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
        if not logger:
            return

        logger.experiment.add_figure(f"recall_filter_matrix/{stage}",
                                     self.draw_confusion_matrix(self.cm_recall),
                                     global_step=epoch)
        self.cm_recall.reset()

        logger.experiment.add_figure(f"precision_filter_matrix/{stage}",
                                self.draw_confusion_matrix(self.cm_precision),
                                global_step=epoch)
        self.cm_precision.reset()
