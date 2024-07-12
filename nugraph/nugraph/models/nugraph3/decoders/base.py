"""NuGraph3 decoder base class"""
from typing import Any, Callable
from abc import ABC
import torch
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics as tm
import matplotlib.pyplot as plt
import seaborn as sn
from ..types import T

class DecoderBase(nn.Module, ABC):
    """NuGraph3 decoder base class"""
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
        self.temp = nn.Parameter(torch.tensor(temperature))
        self.confusion = nn.ModuleDict()

    def arrange(self, batch) -> tuple[T, T]:
        """
        Virtual base class for arranging data

        Args:
            batch: Batch of graph objects
        """
        raise NotImplementedError

    def metrics(self, x: T, y: T, stage: str) -> dict[str, Any]:
        """
        Virtual base class for calculating metrics

        Args:
            x: Model output
            y: Ground truth
            stage: Training stage
        """
        raise NotImplementedError

    def loss(self,
             batch,
             stage: str,
             confusion: bool = False):
        """
        Calculate loss

        Args:
            batch: Batch of graph objects
            stage: Training stage
            confusion: Whether to produce confusion matrices
        """
        x, y = self.arrange(batch)
        w = self.weight * (-1 * self.temp).exp()
        loss = w * self.loss_func(x, y) + self.temp
        metrics = {}
        if stage:
            metrics = self.metrics(x, y, stage)
            metrics[f'loss_{self.name}/{stage}'] = loss
            if stage == 'train':
                metrics[f'temperature/{self.name}'] = self.temp
            if confusion:
                for cm in self.confusion.values():
                    cm.update(x, y)
        return loss, metrics

    def finalize(self, batch) -> None:
        """
        Virtual base class for finalizing model output

        Args:
            batch: Batch of graph objects
        """
        return

    def draw_confusion_matrix(self, cm: tm.ConfusionMatrix) -> plt.Figure:
        """
        Draw confusion matrix

        Args:
            cm: Confusion matrix object
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

    def on_epoch_end(self,
                     logger: TensorBoardLogger,
                     stage: str,
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
        for name, cm in self.confusion.items():
            logger.experiment.add_figure(
                f'{name}/{stage}',
                self.draw_confusion_matrix(cm),
                global_step=epoch)
            cm.reset()
