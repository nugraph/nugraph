from torch import Tensor, cat
import torch.nn as nn

import torchmetrics as tm

import matplotlib.pyplot as plt

from .linear import ClassLinear
from ..util import FocalLoss, RecallLoss

    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 classes: list[str],
                 event_classes: list[str]):
        super().__init__()

        self.name = 'event'
        self.planes = planes
        self.classes = event_classes
        num_planes = len(planes)
        num_classes = len(classes)
        num_features = num_planes * num_classes * node_features

        self.pool = nn.ModuleDict()
        for p in planes:
            self.pool[p] = pyg.nn.aggr.SoftmaxAggregation(learn=True)
        self.net = nn.Sequential(
            nn.Linear(in_features=num_features,
                      out_features=len(event_classes)))

        self.loss_func = FocalLoss(gamma=gamma)
        self.acc_func = tm.Accuracy(task='multiclass',
                                    num_classes=len(event_classes))
        self.cm_true = tm.ConfusionMatrix(task='multiclass',
                                          num_classes=len(event_classes),
                                          normalize='true')
        self.cm_pred = tm.ConfusionMatrix(task='multiclass',
                                          num_classes=len(event_classes),
                                          normalize='pred')

    def forward(self, x: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        return { self.name: { p: self.pool[p](x[p],flatten(1), batch[p])} }

    def loss(self,
             batch,
             name: str,
             confusion: bool = False) -> float:
        metrics = {}
        x = batch['evt'].x
        y = batch['evt'].y
        loss = self.loss_func(x, y)
        metrics[f'event_loss/{name}'] = loss
        acc = 100. * self.acc_func(x, y)
        metrics[f'event_accuracy/{name}'] = acc
        if confusion:
            self.cm_true.update(x, y)
            self.cm_pred.update(x, y)
        return loss, metrics

    def draw_confusion_matrix(self, cm: tm.ConfusionMatrix) -> plt.Figure:
        '''Produce confusion matrix at end of epoch'''
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

    def val_epoch_end(self,
                      logger: 'pl.loggers.TensorBoardLogger',
                      epoch: int) -> None:
        logger.experiment.add_figure('event_efficiency',
                                     self.draw_confusion_matrix(self.cm_true),
                                     global_step=epoch)
        logger.experiment.add_figure('event_purity',
                                     self.draw_confusion_matrix(self.cm_pred),
                                     global_step=epoch)

class SemanticDecoder(nn.Module):
    """NuGraph semantic decoder module.

    Convolve down to a single node score per semantic class for each 2D graph,
    node, and remove intermediate node stores from data object.
    """
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 classes: list[str]):
        super().__init__()

        self.name = 'semantic'
        self.planes = planes
        self.classes = classes
        num_classes = len(classes)

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = ClassLinear(node_features, 1, num_classes)

        self.loss_func = RecallLoss()
        self.acc_func = tm.Accuracy(task='multiclass',
                                    num_classes=num_classes)
        self.acc_func_classwise = tm.Accuracy(task='multiclass',
                                              num_classes=num_classes,
                                              average='none')
        self.cm_true = tm.ConfusionMatrix(task='multiclass',
                                          num_classes=num_classes,
                                          normalize='true')
        self.cm_pred = tm.ConfusionMatrix(task='multiclass',
                                          num_classes=num_classes,
                                          normalize='pred')

    def forward(self, x: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        return { 'x_s': { p: self.net[p](x[p]).squeeze(dim=-1) for p in self.planes } }

    def loss(self,
             batch,
             name: str,
             confusion: bool = False):
        metrics = {}
        x = cat([batch[p].x_s[batch[p].y_f] for p in self.planes], dim=0)
        y = cat([batch[p].y_s for p in self.planes], dim=0)
        loss = self.loss_func(x, y)
        metrics[f'semantic_loss/{name}'] = loss
        acc = 100. * self.acc_func(x, y)
        metrics[f'semantic_accuracy/{name}'] = acc
        for c, a in zip(self.classes, self.acc_func_classwise(x, y)):
            metrics[f'semantic_accuracy_class_{name}/{c}'] = 100. * a
        if confusion:
            self.cm_true.update(x, y)
            self.cm_pred.update(x, y)
        return loss, metrics

    def reset_confusion_matrix(self):
        self.cm_true.reset()
        self.cm_pred.reset()

    def draw_confusion_matrix(self, cm: tm.ConfusionMatrix) -> plt.Figure:
        '''Produce confusion matrix at end of epoch'''
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

    def plot_confusion_matrix(self) -> tuple['plt.Figure']:
        cm_true = self.draw_confusion_matrix(self.cm_true)
        cm_pred = self.draw_confusion_matrix(self.cm_pred)
        return cm_true, cm_pred

    def val_epoch_end(self,
                      logger: 'pl.loggers.TensorBoardLogger',
                      epoch: int) -> None:
        logger.experiment.add_figure('semantic_efficiency',
                                     self.draw_confusion_matrix(self.cm_true),
                                     global_step=epoch)
        logger.experiment.add_figure('semantic_purity',
                                     self.draw_confusion_matrix(self.cm_pred),
                                     global_step=epoch)

class FilterDecoder(nn.Module):
    """NuGraph filter decoder module.

    Convolve down to a single node score, to identify and filter out
    graph nodes that are not part of the primary physics interaction
    """
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 classes: list[str]):
        super().__init__()

        self.name = 'filter'
        self.planes = planes
        self.classes = classes
        num_classes = len(classes)
        num_features = num_classes * node_features

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Linear(num_features, 1)

        self.loss_func = nn.BCELoss()
        self.acc_func = tm.Accuracy(task='binary')
        self.cm_true = tm.ConfusionMatrix(task='binary',
                                          normalize='true')
        self.cm_pred = tm.ConfusionMatrix(task='binary',
                                          normalize='pred')

    def forward(self, x: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        return { self.name: { p: self.net[p](x[p].flatten(start_dim=1)).squeeze(dim=-1) for p in self.planes }}

    def loss(self,
             batch,
             name: str,
             confusion: bool = False) -> float:
        metrics = {}
        x = cat([batch[p].x_f for p in self.planes], dim=0)
        y = cat([batch[p].y_f for p in self.planes], dim=0)
        loss = self.loss_func(x, y.float())
        metrics[f'filter_loss/{name}'] = loss
        acc = 100. * self.acc_func(x, y)
        metrics[f'filter_accuracy/{name}'] = acc
        if confusion:
            self.cm_true.update(x, y)
            self.cm_pred.update(x, y)
        return loss, metrics

    def draw_confusion_matrix(self, cm: tm.ConfusionMatrix) -> plt.Figure:
        '''Produce confusion matrix at end of epoch'''
        confusion = cm.compute().cpu()
        fig = plt.figure(figsize=[8,6])
        sn.heatmap(confusion,
                   xticklabels=['background','signal'],
                   yticklabels=['background','signal'],
                   vmin=0, vmax=1,
                   annot=True)
        plt.ylim(0, len(self.classes))
        plt.xlabel('Assigned label')
        plt.ylabel('True label')
        return fig

    def val_epoch_end(self,
                      logger: 'pl.loggers.TensorBoardLogger',
                      epoch: int) -> None:
        logger.experiment.add_figure('filter_efficiency',
                                     self.draw_confusion_matrix(self.cm_true),
                                     global_step=epoch)
        logger.experiment.add_figure('filter_purity',
                                     self.draw_confusion_matrix(self.cm_pred),
                                     global_step=epoch)