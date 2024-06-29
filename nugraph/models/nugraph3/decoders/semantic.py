"""NuGraph3 semantic decoder"""
from typing import Any
import torch
from torch import nn
from torch_geometric.data import Batch
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import seaborn as sn
import torchmetrics as tm
from ....util import RecallLoss
from ..types import Data

class SemanticDecoder(nn.Module):
    """
    NuGraph3 semantic decoder module

    Convolve planar node embedding down to a set of categorical scores for
    each semantic class.

    Args:
        node_features: Number of planar node features
        planes: List of detector planes
        semantic_classes: List of semantic classes
    """
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 semantic_classes: list[str]):
        super().__init__()

        # loss function
        self.loss = RecallLoss()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.))

        # metrics
        metric_args = {
            "task": "multiclass",
            "num_classes": len(semantic_classes),
            "ignore_index": -1
        }
        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.cm_recall = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.cm_precision = tm.ConfusionMatrix(normalize="pred", **metric_args)

        # network
        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Linear(node_features, len(semantic_classes))

        self.classes = semantic_classes

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 semantic decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        # run network and add output to graph object
        for p, net in self.net.items():
            data[p].x_semantic = net(data[p].x)
            if isinstance(data, Batch):
                data._slice_dict[p]["x_semantic"] = data[p].ptr
                inc = torch.zeros(data.num_graphs, device=data[p].x.device)
                data._inc_dict[p]["x_semantic"] = inc
    
        # calculate loss
        x = torch.cat([data[p].x_semantic for p in self.net], dim=0)
        y = torch.cat([data[p].y_semantic for p in self.net], dim=0)
        w = 2 * (-1 * self.temp).exp()
        loss = w * self.loss(x, y) + self.temp

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"loss_semantic/{stage}"] = loss
            metrics[f"recall_semantic/{stage}"] = self.recall(x, y)
            metrics[f"precision_semantic/{stage}"] = self.precision(x, y)
        if stage == "train":
            metrics["temperature/semantic"] = self.temp
        if stage in ["val", "test"]:
            self.cm_recall.update(x, y)
            self.cm_precision.update(x, y)

        # apply softmax to prediction
        for p in self.net:
            data[p].x_semantic = data[p].x_semantic.softmax(dim=1)

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
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   vmin=0, vmax=1,
                   annot=True)
        plt.ylim(0, len(self.classes))
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

        logger.experiment.add_figure(f"recall_semantic_matrix/{stage}",
                                     self.draw_confusion_matrix(self.cm_recall),
                                     global_step=epoch)
        self.cm_recall.reset()

        logger.experiment.add_figure(f"precision_semantic_matrix/{stage}",
                                self.draw_confusion_matrix(self.cm_precision),
                                global_step=epoch)
        self.cm_precision.reset()
