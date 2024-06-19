"""NuGraph3 semantic decoder"""
from typing import Any
import torch
from torch import nn
import torchmetrics as tm
from .base import DecoderBase
from ....util import RecallLoss
from ..types import T, TD

class SemanticDecoder(DecoderBase):
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
        super().__init__("semantic",
                         planes,
                         semantic_classes,
                         RecallLoss(),
                         weight=2.)

        # torchmetrics arguments
        metric_args = {
            "task": "multiclass",
            "num_classes": len(semantic_classes),
            "ignore_index": -1
        }

        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.confusion["recall_semantic_matrix"] = tm.ConfusionMatrix(
            normalize="true", **metric_args)
        self.confusion["precision_semantic_matrix"] = tm.ConfusionMatrix(
            normalize="pred", **metric_args)

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Linear(node_features, len(semantic_classes))

    def forward(self, x: TD) -> dict[str, TD]:
        """
        NuGraph3 semantic decoder forward pass

        Args:
            x: Node embedding tensor dictionary
        """
        return {"x_semantic": {p: net(x[p]) for p, net in self.net.items()}}

    def arrange(self, batch) -> tuple[T, T]:
        """
        NuGraph3 semantic decoder arrange function

        Args:
            batch: Batch of graph objects
        """
        x = torch.cat([batch[p].x_semantic for p in self.planes], dim=0)
        y = torch.cat([batch[p].y_semantic for p in self.planes], dim=0)
        return x, y

    def metrics(self, x: T, y: T, stage: str) -> dict[str, Any]:
        """
        NuGraph3 semantic decoder metrics function

        Args:
            x: Model output
            y: Ground truth
            stage: Training stage
        """
        return {
            f"recall_semantic/{stage}": self.recall(x, y),
            f"precision_semantic/{stage}": self.precision(x, y)
        }

    def finalize(self, batch) -> None:
        """
        Finalize outputs for NuGraph3 semantic decoder

        Args:
            batch: Batch of graph objects
        """
        for p in self.planes:
            batch[p].s = batch[p].x_semantic.softmax(dim=1)
