from typing import Any, Callable, NoReturn
from argparse import ArgumentParser
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
from pytorch_lightning import LightningModule
import torchmetrics as tm

from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
import seaborn as sn

from ..util import FocalLoss, RecallLoss

PlaneTensor = dict[str, torch.Tensor]

Activation = nn.Tanh

class ClassLinear(nn.Module):
    """Linear convolution module grouped by class, with activation."""
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_classes: int):
        super().__init__()

        self.num_classes = num_classes

        class Linear(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(in_features, out_features)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        self.net = nn.ModuleList([ Linear() for _ in range(num_classes) ])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = torch.tensor_split(X, self.num_classes, dim=1)
        return torch.cat([ net(x[i]) for i, net in enumerate(self.net) ], dim=1)

class PlaneNet(nn.Module):
    '''Module to convolve within each detector plane'''
    def __init__(self,
                 in_features: int,
                 node_features: int,
                 edge_features: int,
                 num_classes: int,
                 planes: list[str],
                 aggr: str = 'add'):
        super().__init__()

        # define individual module block for each plane
        class Net(pyg.nn.MessagePassing):
            def __init__(self):
                super().__init__(node_dim=0, aggr=aggr)

                self.edge_net = nn.Sequential(
                    ClassLinear(2 * (in_features + node_features),
                                edge_features,
                                num_classes),
                    Activation(),
                    ClassLinear(edge_features,
                                1,
                                num_classes),
                    nn.Softmax(dim=1))

                self.node_net = nn.Sequential(
                    ClassLinear(2 * (in_features + node_features),
                                node_features,
                                num_classes),
                    Activation(),
                    ClassLinear(node_features,
                                node_features,
                                num_classes),
                    Activation())

            def forward(self,
                        x: torch.Tensor,
                        edge_index: torch.Tensor):
                return self.propagate(x=x, edge_index=edge_index)

            def message(self, x_i: torch.Tensor, x_j: torch.Tensor):
                return self.edge_net(torch.cat((x_i, x_j), dim=-1).detach()) * x_j

            def update(self, aggr_out: torch.Tensor, x: torch.Tensor):
                return self.node_net(torch.cat((x, aggr_out), dim=-1))

        self.net = nn.ModuleDict({ p: Net() for p in planes })

    def forward(self, x: PlaneTensor, edge_index: PlaneTensor) -> None:
        for p in self.net:
            x[p] = self.net[p](x[p], edge_index[p])

class NexusNet(nn.Module):
    '''Module to project to nexus space and mix detector planes'''
    def __init__(self,
                 in_features: int,
                 node_features: int,
                 edge_features: int,
                 sp_features: int,
                 num_classes: int,
                 planes: list[str],
                 aggr: str = 'mean'):
        super().__init__()

        self.nexus_up = pyg.nn.SimpleConv(node_dim=0)

        self.nexus_net = nn.Sequential(
            ClassLinear(len(planes)*node_features,
                        sp_features,
                        num_classes),
            Activation(),
            ClassLinear(sp_features,
                        sp_features,
                        num_classes),
            Activation())

        class NexusDown(pyg.nn.MessagePassing):
            def __init__(self):
                super().__init__(node_dim=0, aggr=aggr, flow='target_to_source')

                self.edge_net = nn.Sequential(
                    ClassLinear(node_features+sp_features,
                                edge_features,
                                num_classes),
                    Activation(),
                    ClassLinear(edge_features, 1, num_classes),
                    nn.Softmax(dim=1))
                self.node_net = nn.Sequential(
                    ClassLinear(node_features+sp_features,
                                node_features,
                                num_classes),
                    Activation(),
                    ClassLinear(node_features,
                                node_features,
                                num_classes),
                    Activation())

            def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                        n: torch.Tensor) -> torch.Tensor:
                return self.propagate(x=x, n=n, edge_index=edge_index)

            def message(self, x_i: torch.Tensor, n_j: torch.Tensor) -> torch.Tensor:
                return self.edge_net(torch.cat((x_i, n_j), dim=-1).detach()) * n_j

            def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                return self.node_net(torch.cat((x, aggr_out), dim=-1))

        self.nexus_down = nn.ModuleDict({ p: NexusDown() for p in planes })

    def forward(self, x: PlaneTensor, edge_index: PlaneTensor,
                nexus: torch.Tensor) -> None:

        # project up to nexus space
        n = [None] * len(self.nexus_down)
        for i, p in enumerate(self.nexus_down):
            n[i] = self.nexus_up(x=(x[p], nexus), edge_index=edge_index[p])

        # convolve in nexus space
        n = self.nexus_net(torch.cat(n, dim=-1))

        # project back down to planes
        for p in self.nexus_down:
            x[p] = self.nexus_down[p](x=x[p], edge_index=edge_index[p], n=n)

class Encoder(nn.Module):
    """NuGraph2 encoder module.

    Repeat input node features for each class, and then convolve to produce
    initial encoding.
    """
    def __init__(self,
                 in_features: int,
                 node_features: int,
                 planes: list[str],
                 classes: list[str]):
        super().__init__()

        self.planes = planes
        self.num_classes = len(classes)

        def make_net():
            return nn.Sequential(
                ClassLinear(in_features, node_features, self.num_classes),
                Activation())
        self.net = nn.ModuleDict({ p: make_net() for p in planes })

    def forward(self, x: PlaneTensor) -> PlaneTensor:
        return { p: self.net[p](x[p].unsqueeze(1).expand(-1, self.num_classes, -1)) for p in self.planes}

class EventDecoder(nn.Module):
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

    def forward(self, x: PlaneTensor, batch: PlaneTensor) -> dict[str, PlaneTensor]:
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

    def forward(self, x: PlaneTensor, batch: PlaneTensor) -> dict[str, PlaneTensor]:
        return { 'x_s': { p: self.net[p](x[p]).squeeze(dim=-1) for p in self.planes } }

    def loss(self,
             batch,
             name: str,
             confusion: bool = False):
        metrics = {}
        x = torch.cat([batch[p].x_s[batch[p].y_f] for p in self.planes], dim=0)
        y = torch.cat([batch[p].y_s for p in self.planes], dim=0)
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

    def forward(self, x: PlaneTensor, batch: PlaneTensor) -> dict[str, PlaneTensor]:
        return { self.name: { p: self.net[p](x[p].flatten(start_dim=1)).squeeze(dim=-1) for p in self.planes }}

    def loss(self,
             batch,
             name: str,
             confusion: bool = False) -> float:
        metrics = {}
        x = torch.cat([batch[p].x_f for p in self.planes], dim=0)
        y = torch.cat([batch[p].y_f for p in self.planes], dim=0)
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

class NuGraph2(LightningModule):
    """PyTorch Lightning module for model training.

    Wrap the base model in a LightningModule wrapper to handle training and
    inference, and compute training metrics."""
    def __init__(self,
                 in_features: int = 4,
                 node_features: int = 8,
                 edge_features: int = 8,
                 sp_features: int = 8,
                 planes: list[str] = ['u','v','y'],
                 classes: list[str] = ['MIP','HIP','shower','michel','diffuse'],
                 event_classes: list[str] = ['numu','nue','nc'],
                 num_iters: int = 5,
                 event_head: bool = True,
                 semantic_head: bool = True,
                 filter_head: bool = False,
                 checkpoint: bool = False,
                 lr: float = 0.001):
        super().__init__()

        warnings.filterwarnings("ignore", ".*NaN values found in confusion matrix.*")

        self.save_hyperparameters()

        self.planes = planes
        self.classes = classes
        self.event_classes = event_classes
        self.num_iters = num_iters
        self.checkpoint = checkpoint
        self.lr = lr

        self.encoder = Encoder(in_features,
                               node_features,
                               planes,
                               classes)

        self.plane_net = PlaneNet(in_features,
                                  node_features,
                                  edge_features,
                                  len(classes),
                                  planes)

        self.nexus_net = NexusNet(in_features,
                                  node_features,
                                  edge_features,
                                  sp_features,
                                  len(classes),
                                  planes)

        self.decoders = []

        if event_head:
            self.event_decoder = EventDecoder(
                node_features,
                planes,
                classes,
                event_classes)
            self.decoders.append(self.event_decoder)

        if semantic_head:
            self.semantic_decoder = SemanticDecoder(
                node_features,
                planes,
                classes)
            self.decoders.append(self.semantic_decoder)

        if filter_head:
            self.filter_decoder = FilterDecoder(
                node_features,
                planes,
                classes)
            self.decoders.append(self.filter_decoder)

        if len(self.decoders) == 0:
            raise Exception('At least one decoder head must be enabled!')

    def forward(self, x: PlaneTensor, edge_index_plane: PlaneTensor,
                edge_index_nexus: PlaneTensor, nexus: torch.Tensor,
                batch: PlaneTensor) -> PlaneTensor:
        m = self.encoder(x)
        for _ in range(self.num_iters):
            # shortcut connect features
            for i, p in enumerate(self.planes):
                m[p] = torch.cat((m[p], x[p].unsqueeze(1).expand(-1, m[p].size(1), -1)), dim=-1)
            self.plane_net(m, edge_index_plane)
            self.nexus_net(m, edge_index_nexus, nexus)

        ret = {}
        for decoder in self.decoders:
            ret.update(decoder(m, batch))
        return ret

    def step(self, batch):
        x = self(batch.collect('x'),
                 { p: batch[p, 'plane', p].edge_index for p in self.planes },
                 { p: batch[p, 'nexus', 'sp'].edge_index for p in self.planes },
                 torch.empty(batch['sp'].num_nodes, 0),
                 { p: batch[p].batch for p in self.planes })
        for key, value in x.items():
            batch.set_value_dict(key, value)

    def on_train_start(self):
        hpmetrics = { 'max_lr': self.hparams.lr }
        self.logger.log_hyperparams(self.hparams, metrics=hpmetrics)

        scalars = {
            'loss': {'loss': [ 'Multiline', [ 'loss/train', 'loss/val' ]]},
            'acc': {}
        }
        for c in self.classes:
            scalars['acc'][c] = [ 'Multiline', [
                f'semantic_accuracy_class_train/{c}',
                f'semantic_accuracy_class_val/{c}'
            ]]
        self.logger.experiment.add_custom_scalars(scalars)

    def training_step(self,
                      batch,
                      batch_idx: int) -> float:
        self.step(batch)
        total_loss = 0.
        for decoder in self.decoders:
            loss, metrics = decoder.loss(batch, 'train')
            total_loss += loss
            self.log_dict(metrics, batch_size=batch.num_graphs)
        self.log('loss/train', total_loss, batch_size=batch.num_graphs, prog_bar=True)
        return total_loss

    def on_validation_epoch_start(self) -> None:
        for decoder in self.decoders:
            decoder.reset_confusion_matrix()

    def validation_step(self,
                        batch,
                        batch_idx: int) -> None:
        self.step(batch)
        total_loss = 0.
        for decoder in self.decoders:
            loss, metrics = decoder.loss(batch, 'val', True)
            total_loss += loss
            self.log_dict(metrics, batch_size=batch.num_graphs)
        self.log('loss/val', total_loss, batch_size=batch.num_graphs)

    def on_validation_epoch_end(self) -> None:
        for decoder in self.decoders:
            decoder.val_epoch_end(self.logger, self.trainer.current_epoch+1)

    def on_test_epoch_start(self) -> None:
        for decoder in self.decoders:
            decoder.reset_confusion_matrix()

    def test_step(self,
                  batch,
                  batch_idx: int = 0) -> None:
        self(batch)
        for decoder in self.decoders:
            decoder.loss(batch, 'test', True)

    def on_test_epoch_end(self) -> tuple['plt.Figure']:
        for decoder in self.decoders:
            cm_true, cm_pred = decoder.plot_confusion_matrix()
            cm_true.savefig(f'cm_{decoder.name}_true.pdf')
            cm_pred.savefig(f'cm_{decoder.name}_pred.pdf')

    def configure_optimizers(self) -> tuple:
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr)
        onecycle = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], {'scheduler': onecycle, 'interval': 'step'}

    @staticmethod
    def add_model_args(parser: ArgumentParser) -> ArgumentParser:
        '''Add argparse argpuments for model structure'''
        model = parser.add_argument_group('model', 'NuGraph2 model configuration')
        model.add_argument('--node-feats', type=int, default=64,
                           help='Hidden dimensionality of 2D node convolutions')
        model.add_argument('--edge-feats', type=int, default=16,
                           help='Hidden dimensionality of edge convolutions')
        model.add_argument('--sp-feats', type=int, default=16,
                           help='Hidden dimensionality of spacepoint convolutions')
        model.add_argument('--event', action='store_true', default=False,
                           help='Enable event classification head')
        model.add_argument('--semantic', action='store_true', default=False,
                           help='Enable semantic segmentation head')
        model.add_argument('--filter', action='store_true', default=False,
                           help='Enable background filter head')
        return parser

    @staticmethod
    def add_train_args(parser: ArgumentParser) -> ArgumentParser:
        train = parser.add_argument_group('train', 'NuGraph2 training configuration')
        train.add_argument('--no-checkpointing', action='store_true', default=False,
                           help='Disable checkpointing during training')
        train.add_argument('--epochs', type=int, default=80,
                           help='Maximum number of epochs to train for')
        train.add_argument('--learning-rate', type=float, default=0.001,
                           help='Max learning rate during training')
        train.add_argument('--clip-gradients', type=float, default=None,
                           help='Maximum value to clip gradient norm')
        train.add_argument('--gamma', type=float, default=2,
                           help='Focal loss gamma parameter')
        return parser
