"""NuGraph3 semantic decoder"""
from typing import Any
import torch
from torch import nn
from torch_geometric.data import Batch
from pytorch_lightning.loggers import WandbLogger
import wandb
import plotly.express as px
import tempfile
import torchmetrics as tm
from ....util import RecallLoss
from ..types import Data

class SemanticDecoder(nn.Module):
    """
    NuGraph3 semantic decoder module

    Convolve planar node embedding down to a set of categorical scores for
    each semantic class.

    Args:
        hit_features: Number of planar hit node features
        semantic_classes: List of semantic classes
    """
    def __init__(self,
                 hit_features: int,
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
        self.net = nn.Linear(hit_features, len(semantic_classes))

        self.classes = semantic_classes

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 semantic decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        # run network and add output to graph object
        data["hit"].x_semantic = self.net(data["hit"].x)
        if isinstance(data, Batch):
            data._slice_dict["hit"]["x_semantic"] = data["hit"].ptr
            inc = torch.zeros(data.num_graphs, device=data["hit"].x.device)
            data._inc_dict["hit"]["x_semantic"] = inc
    
        # calculate loss
        x = data["hit"].x_semantic
        y = data["hit"].y_semantic
        w = 2 * (-1 * self.temp).exp()
        loss = w * self.loss(x, y) + self.temp

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"semantic/loss-{stage}"] = loss
            metrics[f"semantic/recall-{stage}"] = self.recall(x, y)
            metrics[f"semantic/precision-{stage}"] = self.precision(x, y)
        if stage == "train":
            metrics["semantic/temperature"] = self.temp
        if stage in ["val", "test"]:
            self.cm_recall.update(x, y)
            self.cm_precision.update(x, y)

        # apply softmax to prediction
        data["hit"].x_semantic = data["hit"].x_semantic.softmax(dim=1)

        return loss, metrics

    def draw_matrix(self, cm: tm.ConfusionMatrix, label: str) -> wandb.Table:
        """
        Draw confusion matrix

        Args:
            cm: Confusion matrix object
        """
        confusion = cm.compute().cpu()
        table = wandb.Table(columns=["plotly_figure"])
        fig = px.imshow(
            confusion, zmax=1, text_auto=True,
            labels=dict(x="Predicted", y="True", color=label),
            x=self.classes, y=self.classes)
        with tempfile.NamedTemporaryFile() as f:
            fig.write_html(f.name, auto_play=False)
            table.add_data(wandb.Html(f.name))
        return table

    def on_epoch_end(self, logger: WandbLogger, stage: str,
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

        table = self.draw_matrix(self.cm_recall, "Recall")
        wandb.log({f"semantic/recall-matrix-{stage}": table})
        self.cm_recall.reset()

        table = self.draw_matrix(self.cm_precision, "Precision")
        wandb.log({f"semantic/precision-matrix-{stage}": table})
        self.cm_precision.reset()
