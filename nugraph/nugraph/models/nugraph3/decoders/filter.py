"""NuGraph3 filter decoder"""
from typing import Any
import torch
from torch import nn
import torchmetrics as tm
from .base import DecoderBase
from ..types import T, TD

class FilterDecoder(DecoderBase):
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
        super().__init__('filter',
                         planes,
                         ('noise', 'signal'),
                         nn.BCELoss(),
                         weight=2.)

        # torchmetrics arguments
        metric_args = {
            'task': 'binary'
        }

        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.confusion['recall_filter_matrix'] = tm.ConfusionMatrix(
            normalize='true', **metric_args)
        self.confusion['precision_filter_matrix'] = tm.ConfusionMatrix(
            normalize='pred', **metric_args)

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Sequential(
                nn.Linear(node_features, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: TD) -> dict[str, TD]:
        """
        NuGraph3 filter decoder forward pass

        Args:
            x: Node embedding tensor dictionary
        """
        return {"x_filter": {p: net(x[p]).squeeze(dim=-1) for p, net in self.net.items()}}

    def arrange(self, batch: TD) -> tuple[T, T]:
        """
        NuGraph3 filter decoder arrange function

        Args:
            batch: Batch of graph objects
        """
        x = torch.cat([batch[p].x_filter for p in self.planes], dim=0)
        y = torch.cat([(batch[p].y_semantic!=-1).float() for p in self.planes], dim=0)
        return x, y

    def metrics(self, x: T, y: T, stage: str) -> dict[str, Any]:
        """
        NuGraph3 filter decoder metrics function

        Args:
            x: Model output
            y: Ground truth
            stage: Training stage
        """
        return {
            f'recall_filter/{stage}': self.recall(x, y),
            f'precision_filter/{stage}': self.precision(x, y)
        }
