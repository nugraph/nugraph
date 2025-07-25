"""NuGraph3 event decoder"""
from typing import Any
import tempfile
import torch
from torch import nn
import torchmetrics as tm
from torch_geometric.data import Batch
from pytorch_lightning.loggers import Logger
from ....util import ConfusionMatrixLogger, RecallLoss
from ..types import Data

class EventDecoder(nn.Module):
    """
    NuGraph3 event decoder module

    Convolve interaction node embedding down to a set of categorical scores
    for each event class.

    Args:
        interaction_features: Number of interaction node features
        event_classes: List of event classes
    """
    def __init__(self,
                 interaction_features: int,
                 event_classes: list[str]):
        super().__init__()

        # loss function
        self.loss = RecallLoss()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.))

        # metrics
        metric_args = {
            "task": "multiclass",
            "num_classes": len(event_classes)
        }
        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.cm_logger = ConfusionMatrixLogger(event_classes)
        self.cm_recall = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.cm_precision = tm.ConfusionMatrix(normalize="pred", **metric_args)

        # network
        self.net = nn.Linear(in_features=interaction_features,
                             out_features=len(event_classes))

        self.classes = event_classes

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 event decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        # run network and calculate loss
        x = self.net(data["evt"].x)
        y = data["evt"].y
        w = 2 * (-1 * self.temp).exp()
        loss = w * self.loss(x, y) + self.temp

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"event/loss-{stage}"] = loss
            metrics[f"event/recall-{stage}"] = self.recall(x, y)
            metrics[f"event/precision-{stage}"] = self.precision(x, y)
        if stage == "train":
            metrics["temperature/event"] = self.temp
        if stage in ["val", "test"]:
            self.cm_recall.update(x, y)
            self.cm_precision.update(x, y)

        # add inference output to graph object
        data["evt"].e = x.softmax(dim=1)
        if isinstance(data, Batch):
            # pylint: disable=protected-access
            data._slice_dict["evt"]["e"] = data["evt"].ptr
            inc = torch.zeros(data.num_graphs, device=data["evt"].x.device)
            data._inc_dict["evt"]["e"] = inc

        return loss, metrics

    def on_epoch_end(self, logger: Logger | list[Logger], stage: str,
                     epoch: int) -> None: # pylint: disable=unused-argument
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: PyTorch Lightning logger object(s)
            stage: Training stage
            epoch: Training epoch index
        """
        self.cm_logger.log(f"event/recall-matrix-{stage}",
                           self.cm_recall, logger, epoch)
        self.cm_logger.log(f"event/precision-matrix-{stage}",
                           self.cm_precision, logger, epoch)
