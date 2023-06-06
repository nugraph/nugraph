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

class Message2D(pyg.nn.MessagePassing):
    def __init__(self,
                 in_features: int,
                 node_features: int,
                 edge_features: int,
                 num_classes: int,
                 aggr: str = 'add'):
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

class SPNet(nn.Module):
    """Module for propagating global features between planes.

    Propagate features from 2D nodes up to 3D nodes using 2D-to-3D edges,
    convolve features of 3D nodes, then form attention weights on 2D-to-3D
    edges and propagate 3D node features back to 2D nodes using these weights.
    Skip-connect with original 2D features and convolve once more.
    """
    def __init__(self,
                 node_features: int,
                 edge_features: int,
                 sp_features: int,
                 planes: list[str],
                 classes: list[str],
                 checkpoint: bool):
        super().__init__()

        self.planes = planes
        self.checkpoint = checkpoint

        num_planes = len(planes)
        num_classes = len(classes)

        self.node_net_3d = nn.Sequential(
            ClassLinear(num_planes*node_features,
                        sp_features,
                        num_classes),
            Activation(),
            ClassLinear(sp_features,
                        sp_features,
                        num_classes),
            Activation())

        self.edge_net = nn.ModuleDict({
            p: nn.Sequential(
                ClassLinear(node_features+sp_features,
                            edge_features,
                            num_classes),
                Activation(),
                ClassLinear(edge_features, 1, num_classes),
                nn.Softmax(dim=1))
            for p in planes
            })

        self.node_net_2d = nn.ModuleDict({
            p: nn.Sequential(
                ClassLinear(node_features+sp_features,
                            node_features,
                            num_classes),
                Activation(),
                ClassLinear(node_features,
                            node_features,
                            num_classes),
                Activation())
            for p in planes
            })

    def sp_to_hit(self,
                  m_2d: torch.Tensor,
                  m_3d: torch.Tensor,
                  edge_index: torch.Tensor,
                  plane: str) -> torch.Tensor:
        hit, sp = edge_index
        edge_feats = torch.cat([m_2d[hit], m_3d[sp]], dim=-1).detach()
        return self.edge_net[plane](edge_feats)

    def forward(self, data) -> None:

        # propagate 2D hit features to 3D and convolve
        # TODO: let's just do this with torch.new_empty in future
        def hit_to_sp(data,
                      plane: str) -> torch.Tensor:
            hit, sp = data[plane, 'nexus', 'sp'].edge_index
            return pyg.utils.scatter(data[plane].m[hit],
                                     sp,
                                     dim=0,
                                     dim_size=data['sp'].num_nodes,
                                     reduce='add')
        m_sp = torch.cat([hit_to_sp(data, p) for p in self.planes], dim=-1)
        if self.checkpoint and self.training:
            m_sp = checkpoint(self.node_net_3d, m_sp)
        else:
            m_sp = self.node_net_3d(m_sp)

        # merge 3D hit features back down to 2D
        for p in self.planes:
            if self.checkpoint and self.training:
                edge_feats = checkpoint(self.sp_to_hit,
                                        data[p].m,
                                        m_sp,
                                        data[p, 'nexus', 'sp'].edge_index,
                                        p)
            else:
                edge_feats = self.sp_to_hit(data[p].m, m_sp, data[p, 'nexus', 'sp'].edge_index, p)

            hit, sp = data[p, 'nexus', 'sp'].edge_index
            m_hit = pyg.utils.scatter(edge_feats*m_sp[sp],
                                      hit,
                                      dim=0,
                                      dim_size=data[p].num_nodes,
                                      reduce='mean')
            m = torch.cat([data[p].m, m_hit], dim=-1)
            if self.checkpoint and self.training:
                data[p].m = checkpoint(self.node_net_2d[p], m)
            else:
                data[p].m = self.node_net_2d[p](m)

class MessageNet(nn.Module):
    """Message-passing backbone of the NuGraph2 model.

    Apply edge network to form classwise edge attention weights, use those
    weights to pass node features across graph edges and then convolve 2D node
    features. Then project 2D graph node features into a 3D node space,
    convolve again, form 2D-to-3D edge attention weights and then use those
    weights to propagate 3D information back down to 2D nodes. This entire
    procedure is applied iteratively multiple times, to propagate information
    throughout the graph.
    """
    def __init__(self,
                 in_features: int,
                 node_features: int,
                 edge_features: int,
                 sp_features: int,
                 planes: list[str],
                 classes: list[str],
                 num_iters: int,
                 checkpoint: bool):
        super().__init__()

        self.planes = planes
        self.num_iters = num_iters
        self.checkpoint = checkpoint

        num_planes = len(planes)
        num_classes = len(classes)

        def make_net2d():
            return Message2D(in_features,
                             node_features,
                             edge_features,
                             num_classes)
        self.net2d = nn.ModuleList([make_net2d() for _ in planes])

        self.sp_net = SPNet(node_features,
                            edge_features,
                            sp_features,
                            planes,
                            classes,
                            checkpoint)

    def forward(self, data) -> None:
        for _ in range(self.num_iters):
            for i, p in enumerate(self.planes):
                num_classes = data[p].m.size(-2)
                S = data[p].x.unsqueeze(1).expand(-1, num_classes, -1)
                M = torch.cat((data[p].m, S), dim=-1)
                E = data[p, 'plane', p].edge_index
                if self.checkpoint and self.training:
                    M = checkpoint(self.net2d[i], M, E)
                else:
                    M = self.net2d[i](M, E)
                data[p].m = M
            self.sp_net(data)

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

    def forward(self, batch: pyg.data.HeteroData) -> None:
        for p in self.planes:
            x = batch[p].x.unsqueeze(1).expand(-1, self.num_classes, -1)
            batch[p].m = self.net[p](x)

class EventDecoder(nn.Module):
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 classes: list[str],
                 event_classes: list[str],
                 gamma: float = 2):
        super().__init__()

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

    def forward(self, data) -> None:
        x = []
        for p in self.planes:
            x.append(self.pool[p](data[p].m.flatten(1), data[p].batch))
        data['evt'].x = self.net(torch.cat(x, dim=-1))

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
                 classes: list[str],
                 weight: torch.Tensor = None):
        super().__init__()

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

    def forward(self, data) -> None:
        for p in self.planes:
            data[p].x_s = self.net[p](data[p].m).squeeze(dim=-1)

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

    def forward(self, data):
        for p in self.planes:
            x = self.net[p](data[p].m.flatten(start_dim=1)).squeeze(dim=-1)
            data[p].x_f = torch.sigmoid(x.clamp(-1000., 1000.))

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
                 lr: float = 0.001,
                 semantic_weight: torch.Tensor = None,
                 gamma: float = 2):
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

        self.message_net = MessageNet(in_features,
                                      node_features,
                                      edge_features,
                                      sp_features,
                                      planes,
                                      classes,
                                      num_iters,
                                      checkpoint)

        self.decoders = []

        if event_head:
            self.event_decoder = EventDecoder(
                node_features,
                planes,
                classes,
                event_classes,
                gamma)
            self.decoders.append(self.event_decoder)

        if semantic_head:
            self.semantic_decoder = SemanticDecoder(
                node_features,
                planes,
                classes,
                semantic_weight)
            self.decoders.append(self.semantic_decoder)

        if filter_head:
            self.filter_decoder = FilterDecoder(
                node_features,
                planes,
                classes)
            self.decoders.append(self.filter_decoder)

        if len(self.decoders) == 0:
            raise Exception('At least one decoder head must be enabled!')

    def forward(self, data) -> None:
        for p in self.planes:
            data[p].x.requires_grad_()
        self.encoder(data)
        self.message_net(data)
        for decoder in self.decoders:
            decoder(data)

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
        self(batch)
        total_loss = 0.
        for decoder in self.decoders:
            loss, metrics = decoder.loss(batch, 'train')
            total_loss += loss
            self.log_dict(metrics, batch_size=batch.num_graphs)
        self.log('loss/train', total_loss, batch_size=batch.num_graphs, prog_bar=True)
        return total_loss

    def validation_step(self,
                        batch,
                        batch_idx: int) -> NoReturn:
        self(batch)
        total_loss = 0.
        for decoder in self.decoders:
            loss, metrics = decoder.loss(batch, 'val', True)
            total_loss += loss
            self.log_dict(metrics, batch_size=batch.num_graphs)
        self.log('loss/val', total_loss, batch_size=batch.num_graphs)

    def on_validation_epoch_end(self) -> NoReturn:
        for decoder in self.decoders:
            decoder.val_epoch_end(self.logger, self.trainer.current_epoch+1)

    def test_step(self,
                  batch,
                  batch_idx: int = 0) -> None:
        self(batch)

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
