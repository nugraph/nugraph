"""NuGraph3 vertex decoder"""
from typing import Any
from torch import nn
from ....util import LogCoshLoss
from .base import DecoderBase
from ..types import T, TD

class VertexDecoder(DecoderBase):
    """
    NuGraph3 vertex decoder module

    Convolve interaction node embedding down to a set of 3D coordinates
    predicting the location of the neutrino interaction vertex.

    Args:
        interaction_features: Number of interaction node features
        planes: List of detector planes
        semantic_classes: List of semantic classes
    """
    def __init__(self,
                 interaction_features: int,
                 planes: list[str],
                 semantic_classes: list[str]):
        super().__init__("vertex",
                         planes,
                         semantic_classes,
                         LogCoshLoss(),
                         weight=1.,
                         temperature=5.)

        self.net = nn.Linear(interaction_features, 3)

    def forward(self, x: TD) -> dict[str, TD]:
        """
        NuGraph3 vertex decoder forward pass

        Args:
            x: Node embedding tensor dictionary
        """
        return {"v": {"evt": self.net(x["evt"])}}

    def arrange(self, batch) -> tuple[T, T]:
        """
        NuGraph3 vertex decoder arrange function

        Args:
            batch: Batch of graph objects
        """
        x = batch["evt"].v
        y = batch["evt"].y_vtx
        return x, y

    def metrics(self, x: T, y: T, stage: str) -> dict[str, Any]:
        """
        NuGraph3 vertex decoder metrics function

        Args:
            x: Model output
            y: Ground truth
            stage: Training stage
        """
        xyz = (x-y).abs().mean(dim=0)
        return {
            f"vertex-resolution-x/{stage}": xyz[0],
            f"vertex-resolution-y/{stage}": xyz[1],
            f"vertex-resolution-z/{stage}": xyz[2],
            f"vertex-resolution/{stage}": xyz.square().sum().sqrt()
        }
