"""NuGraph3 Michel filter decoder"""
from typing import Any

import torch
from torch import nn
import torchmetrics as tm
from torch_geometric.data import Batch
from pytorch_lightning.loggers import Logger

from nugraph.util import ConfusionMatrixLogger
from ..types import Data


class MichelFilterDecoder(nn.Module):
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

        # Store Michel semantic class index
        self.michel_label = michel_label

        # Metrics
        metric_args = {"task": "binary"}
        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        #self.cm_logger = ConfusionMatrixLogger(("noise", "signal"))
        self.cm_logger = ConfusionMatrixLogger(("non_michel", "michel"))
        self.cm_recall = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.cm_precision = tm.ConfusionMatrix(normalize="pred", **metric_args)

        # Network: one logit per hit
        self.net = nn.Linear(hit_features, 1)

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        Forward pass for Michel filter decoder.
        """

        # One raw logit per hit
        x = self.net(data["hit"].x).squeeze(dim=-1)

        # Use existing semantic labels
        semantic = data["hit"].y_semantic

        # Ignore unlabeled / invalid hits
        mask = semantic != -1

        # Binary target:
        #   1 = Michel semantic hit
        #   0 = non-Michel semantic hit
        target = (semantic == self.michel_label).float()

        x_masked = x[mask]
        y = target[mask]

        # Loss weighting, following the default filter style
        w = 2 * (-1 * self.temp).exp()

        if mask.any():
            loss = w * self.loss(x_masked, y) + self.temp
        else:
            # Safe zero loss if a batch somehow has no valid labels
            loss = x.sum() * 0.0

        metrics = {}

        if stage:
            metrics[f"michel_filter/loss-{stage}"] = loss

        if stage and mask.any():
            metrics[f"michel_filter/recall-{stage}"] = self.recall(x_masked, y)
            metrics[f"michel_filter/precision-{stage}"] = self.precision(x_masked, y)

        if stage == "train":
            metrics["temperature/michel_filter"] = self.temp

        if stage in ["val", "test"] and mask.any():
            self.cm_recall.update(x_masked, y)
            self.cm_precision.update(x_masked, y)

        # Store Michel probability on all hits
        data["hit"].x_michel_filter = x.sigmoid()

        if isinstance(data, Batch):
            data._slice_dict["hit"]["x_michel_filter"] = data["hit"].ptr
            inc = torch.zeros(data.num_graphs, device=data["hit"].x.device)
            data._inc_dict["hit"]["x_michel_filter"] = inc

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

        self.cm_logger.log(
            f"michel_filter/recall-matrix-{stage}",
            self.cm_recall,
            logger,
            epoch,
        )

        self.cm_logger.log(
            f"michel_filter/precision-matrix-{stage}",
            self.cm_precision,
            logger,
            epoch,
        )
