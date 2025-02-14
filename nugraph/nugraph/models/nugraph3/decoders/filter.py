"""NuGraph3 filter decoder"""
from typing import Any
import tempfile
import torch
from torch import nn
import torchmetrics as tm
from torch_geometric.data import Batch
from pytorch_lightning.loggers import WandbLogger
import wandb
import plotly.express as px
from ..types import Data

class FilterDecoder(nn.Module):
    """
    NuGraph3 filter decoder module

    Convolve hit node embedding down to a single node score to identify and
    filter out graph nodes that are not part of the primary physics
    interaction.

    Args:
        hit_features: Number of hit node features
    """
    def __init__(self, hit_features: int):
        super().__init__()

        # loss function
        self.loss = nn.BCEWithLogitsLoss()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.))

        # metrics
        metric_args = {"task": "binary"}
        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.cm_recall = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.cm_precision = tm.ConfusionMatrix(normalize="pred", **metric_args)

        # network
        self.net = nn.Linear(hit_features, 1)

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 filter decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        # calculate loss
        x = self.net(data["hit"].x).squeeze(dim=-1)
        y = (data["hit"].y_semantic != -1).float()
        w = 2 * (-1 * self.temp).exp()
        loss = w * self.loss(x, y) + self.temp

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"filter/loss-{stage}"] = loss
            metrics[f"filter/recall-{stage}"] = self.recall(x, y)
            metrics[f"filter/precision-{stage}"] = self.precision(x, y)
        if stage == "train":
            metrics["temperature/filter"] = self.temp
        if stage in ["val", "test"]:
            self.cm_recall.update(x, y)
            self.cm_precision.update(x, y)

        # run network and add output to graph object
        data["hit"].x_filter = x.sigmoid()
        if isinstance(data, Batch):
            # pylint: disable=protected-access
            data._slice_dict["hit"]["x_filter"] = data["hit"].ptr
            inc = torch.zeros(data.num_graphs, device=data["hit"].x.device)
            data._inc_dict["hit"]["x_filter"] = inc

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
            x=("noise", "signal"), y=("noise", "signal"))
        with tempfile.NamedTemporaryFile() as f:
            fig.write_html(f.name, auto_play=False)
            table.add_data(wandb.Html(f.name))
        return table

    def on_epoch_end(self, logger: WandbLogger, stage: str,
                     epoch: int) -> None: # pylint: disable=unused-argument
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: Wandb logger object
            stage: Training stage
            epoch: Training epoch index
        """
        if not logger:
            return

        table = self.draw_matrix(self.cm_recall, "Recall")
        wandb.log({f"filter/recall-matrix-{stage}": table})
        self.cm_recall.reset()

        table = self.draw_matrix(self.cm_precision, "Precision")
        wandb.log({f"filter/precision-matrix-{stage}": table})
        self.cm_precision.reset()
