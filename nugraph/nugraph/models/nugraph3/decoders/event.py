"""NuGraph3 event decoder"""
from typing import Any
from torch import nn
import torchmetrics as tm
from .base import DecoderBase
from ....util import RecallLoss
from ..types import T, TD

class EventDecoder(DecoderBase):
    """
    NuGraph3 event decoder module

    Convolve interaction node embedding down to a set of categorical scores
    for each event class.

    Args:
        interaction_features: Number of interaction node features
        planes: List of detector planes
        event_classes: List of event classes
    """
    def __init__(self,
                 interaction_features: int,
                 planes: list[str],
                 event_classes: list[str]):
        super().__init__("event",
                         planes,
                         event_classes,
                         RecallLoss(),
                         weight=2.)

        # torchmetrics arguments
        metric_args = {
            "task": "multiclass",
            "num_classes": len(event_classes)
        }

        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.confusion["recall_event_matrix"] = tm.ConfusionMatrix(
            normalize="true", **metric_args)
        self.confusion["precision_event_matrix"] = tm.ConfusionMatrix(
            normalize="pred", **metric_args)

        self.net = nn.Linear(in_features=interaction_features,
                             out_features=len(event_classes))

    def forward(self, x: TD) -> dict[str, TD]:
        """
        NuGraph3 event decoder forward pass

        Args:
            x: Node embedding tensor dictionary
        """
        return {"e": {"evt": self.net(x["evt"])}}

    def arrange(self, batch) -> tuple[T, T]:
        """
        NuGraph3 event decoder arrange function

        Args:
            batch: Batch of graph objects
        """
        return batch["evt"].e, batch["evt"].y

    def metrics(self, x: T, y: T, stage: str) -> dict[str, Any]:
        """
        NuGraph3 event decoder metrics function

        Args:
            x: Model output
            y: Ground truth
            stage: Training stage
        """
        return {
            f"recall_event/{stage}": self.recall(x, y),
            f"precision_event/{stage}": self.precision(x, y)
        }

    def finalize(self, batch) -> None:
        """
        Finalize outputs for NuGraph3 event decoder

        Args:
            batch: Batch of graph objects
        """
        batch["evt"].e = batch["evt"].e.softmax(dim=1)
