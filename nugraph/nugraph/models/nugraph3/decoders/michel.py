"""NuGraph3 Michel filter decoder"""
from typing import Any

import torch
from torch import nn
import torchmetrics as tm
from torch_geometric.data import Batch
from pytorch_lightning.loggers import Logger

from nugraph.util import ConfusionMatrixLogger
from ..types import Data


class MichelDecoder(nn.Module):
    """
    NuGraph3 Michel filter decoder.

    This decoder predicts a binary score per hit:

        1 -> Michel hit
        0 -> non-Michel hit

    Michel semantic hits vs all other labeled hits.
    """

    def __init__(self, hit_features: int, michel_label: int = 5):
        super().__init__()

        # loss function
        self.loss = nn.BCEWithLogitsLoss()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.0))

        # store michel semantic class index
        self.michel_label = michel_label

        # metrics
        metric_args = {"task": "binary"}
        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.f1 = tm.F1Score(**metric_args)
        self.cm = tm.ConfusionMatrix(**metric_args)
        self.cm_logger = ConfusionMatrixLogger(("background", "michel"))

        # network
        self.net = nn.Linear(hit_features, 1)

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        Forward pass for Michel filter decoder.
        """

        h = data["hit"]

        # calculate loss
        x = self.net(h.x).squeeze(dim=-1)
        y = (h.y_semantic == self.michel_label).float()
        w = 2 * (-1 * self.temp).exp()
        loss = w * self.loss(x, y) + self.temp

        metrics = {}

        if stage:
            metrics[f"michel/loss-{stage}"] = loss
            metrics[f"michel/recall-{stage}"] = self.recall(x, y)
            metrics[f"michel/precision-{stage}"] = self.precision(x, y)
            metrics[f"michel/f1-{stage}"] = self.f1(x, y)

        if stage == "train":
            metrics["temperature/michel"] = self.temp

        if stage in ["val", "test"]:
            self.cm.update(x, y)

        # Store Michel probability on all hits
        h.x_michel = x.sigmoid()

        if isinstance(data, Batch):
            data._slice_dict["hit"]["x_michel"] = h.ptr
            inc = torch.zeros(data.num_graphs, device=h.x.device)
            data._inc_dict["hit"]["x_michel"] = inc

        return loss, metrics

    def on_epoch_end(
        self,
        logger: Logger | list[Logger],
        stage: str,
        epoch: int,
    ) -> None:
        """
        End-of-epoch callback for confusion matrix logging.
        """
        self.cm_logger.log("michel", stage, self.cm, logger, epoch)
