"""NuGraph2 decoders"""
from typing import Any, Callable
from abc import ABC
import tempfile

from torch import Tensor, tensor, cat
import torch.nn as nn

import torchmetrics as tm
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger

import matplotlib.pyplot as plt
import seaborn as sn
import plotly.express as px

import wandb

from .linear import ClassLinear
from ...util import RecallLoss

class DecoderBase(nn.Module, ABC):
    '''Base class for all NuGraph decoders'''
    def __init__(self,
                 name: str,
                 planes: list[str],
                 classes: list[str],
                 loss_func: Callable,
                 weight: float,
                 temperature: float = 0.):
        super().__init__()
        self.name = name
        self.planes = planes
        self.classes = classes
        self.loss_func = loss_func
        self.weight = weight
        self.temp = nn.Parameter(tensor(temperature))
        self.confusion = nn.ModuleDict()

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        raise NotImplementedError

    def loss(self,
             batch,
             stage: str,
             confusion: bool = False):
        x, y = self.arrange(batch)
        metrics = self.metrics(x, y, stage)
        w = self.weight * (-1 * self.temp).exp()
        loss = w * self.loss_func(x, y) + self.temp
        metrics[f'loss_{self.name}/{stage}'] = loss
        if stage == 'train':
            metrics[f'temperature/{self.name}'] = self.temp
        for cm in self.confusion.values():
            cm.update(x, y)
        return loss, metrics

    def draw_matrix_tensorboard(self, cm: tm.ConfusionMatrix) -> plt.Figure:
        """
        Draw confusion matrix for tensorboard logging
        
        Args:
            cm: Confusion matrix
        """
        confusion = cm.compute().cpu()
        fig = plt.figure(figsize=[8,6])
        sn.heatmap(confusion,
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   vmin=0, vmax=1,
                   annot=True)
        plt.ylim(0, len(self.classes))
        plt.xlabel('Assigned label')
        plt.ylabel('True label')
        return fig

    def draw_matrix_wandb(self, cm: tm.ConfusionMatrix, label: str) -> wandb.Table:
        """
        Draw confusion matrix for wandb logging
        
        Args:
            cm: Confusion matrix
            label: Confusion matrix label
        """
        confusion = cm.compute().cpu()
        table = wandb.Table(columns=["plotly_figure"])
        fig = px.imshow(
            confusion, zmax=1, text_auto=True,
            labels={"x": "Predicted", "y": "True", "color": label},
            x=self.classes, y=self.classes)
        with tempfile.NamedTemporaryFile() as f:
            fig.write_html(f.name, auto_play=False)
            table.add_data(wandb.Html(f.name))
        return table


    def on_epoch_end(self, logger: Logger, stage: str, epoch: int) -> None:
        """
        End-of-epoch decoder callback for logging confusion matrices

        Args:
            logger: Logger instance
            stage: Name of current stage
            epoch: Epoch number
        """
        if isinstance(logger, TensorBoardLogger):
            for name, cm in self.confusion.items():
                logger.experiment.add_figure(
                    f'{name}/{stage}',
                    self.draw_matrix_tensorboard(cm),
                    global_step=epoch)
                cm.reset()

        if isinstance(logger, WandbLogger):
            cm_recall, cm_precision = self.confusion.values()

            table = self.draw_matrix_wandb(cm_recall, "Recall")
            wandb.log({f"semantic/recall-matrix-{stage}": table})
            cm_recall.reset()

            table = self.draw_matrix_wandb(cm_precision, "Precision")
            wandb.log({f"semantic/precision-matrix-{stage}": table})
            cm_precision.reset()

class SemanticDecoder(DecoderBase):
    """NuGraph semantic decoder module.

    Convolve down to a single node score per semantic class for each 2D graph,
    node, and remove intermediate node stores from data object.
    """
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 semantic_classes: list[str]):
        super().__init__('semantic',
                         planes,
                         semantic_classes,
                         RecallLoss(),
                         weight=2.)

        # torchmetrics arguments
        metric_args = {
            'task': 'multiclass',
            'num_classes': len(semantic_classes),
            'ignore_index': -1
        }

        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.confusion['recall_semantic_matrix'] = tm.ConfusionMatrix(
            normalize='true', **metric_args)
        self.confusion['precision_semantic_matrix'] = tm.ConfusionMatrix(
            normalize='pred', **metric_args)

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = ClassLinear(node_features, 1, len(semantic_classes))

    def forward(self, x: dict[str, Tensor],
                batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        return { 'x_semantic': { p: self.net[p](x[p]).squeeze(dim=-1) for p in self.planes } }

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        x = cat([batch[p].x_semantic for p in self.planes], dim=0)
        y = cat([batch[p].y_semantic for p in self.planes], dim=0)
        return x, y

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        return {
            f'recall_semantic/{stage}': self.recall(x, y),
            f'precision_semantic/{stage}': self.precision(x, y)
        }

class FilterDecoder(DecoderBase):
    """NuGraph filter decoder module.

    Convolve down to a single node score, to identify and filter out
    graph nodes that are not part of the primary physics interaction
    """
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 semantic_classes: list[str]):
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

        num_features = len(semantic_classes) * node_features
        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Sequential(
                nn.Linear(num_features, 1),
                nn.Sigmoid())

    def forward(self, x: dict[str, Tensor],
                batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        return { 'x_filter': { p: self.net[p](x[p].flatten(start_dim=1)).squeeze(dim=-1) for p in self.planes }}

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        x = cat([batch[p].x_filter for p in self.planes], dim=0)
        y = cat([(batch[p].y_semantic!=-1).float() for p in self.planes], dim=0)
        return x, y

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        return {
            f'recall_filter/{stage}': self.recall(x, y),
            f'precision_filter/{stage}': self.precision(x, y)
        }
