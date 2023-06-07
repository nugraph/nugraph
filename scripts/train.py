#!/usr/bin/env python

import sys
import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import LearningRateMonitor
import nugraph as ng
import signal

import warnings
warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')

Data = ng.data.H5DataModule
Model = ng.models.NuGraph2

def configure():
    parser = argparse.ArgumentParser(sys.argv[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--name', type=str, default=None,
                       help='Training instance name, for logging purposes')
    group.add_argument('--resume', type=str, default=None,
                       help='Checkpoint file to resume training from')
    parser.add_argument('--devices', nargs='+', type=int, default=None,
                        help='List of devices to train with')
    parser.add_argument('--logdir', type=str, default='auto',
                        help='Output directory to write logs to')
    parser.add_argument('--profiler', type=str, default=None,
                        help='Enable requested profiler')
    parser.add_argument('--detect-anomaly', action='store_true', default=False,
                        help='Enable PyTorch anomaly detection')
    parser = Data.add_data_args(parser)
    parser = Model.add_model_args(parser)
    parser = Model.add_train_args(parser)
    return parser.parse_args()

def train(args):

    torch.manual_seed(1)

    # Load dataset
    nudata = Data(args.data_path, batch_size=args.batch_size)

    logdir = args.logdir
    if logdir == 'auto':
        logdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')

    # are we resuming an existing training?
    resume = args.resume is not None
    if resume:
        model = Model.load_from_checkpoint(args.resume)
        stub = os.path.dirname(os.path.dirname(args.resume))
        stub, version = os.path.split(stub)
        name = os.path.basename(stub)
    else:
        model = Model(in_features=4,
                      node_features=args.node_feats,
                      edge_features=args.edge_feats,
                      sp_features=args.sp_feats,
                      planes=nudata.planes,
                      classes=nudata.classes,
                      num_iters=5,
                      event_head=args.event,
                      semantic_head=args.semantic,
                      filter_head=args.filter,
                      checkpoint=not args.no_checkpointing,
                      lr=args.learning_rate,
                      semantic_weight=weights,
                      gamma=args.gamma)
        name = args.name
        version = None
        os.makedirs(os.path.join(logdir, args.name), exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=logdir,
                                          name=name, version=version,
                                          default_hp_metric=False)

    callbacks = [
        LearningRateMonitor(logging_interval='step')
    ]

    plugins = [
        SLURMEnvironment(requeue_signal=signal.SIGHUP)
    ]

    if args.devices is None:
        print('No devices specified – training with CPU')

    accelerator = 'cpu' if args.devices is None else 'gpu'
    trainer = pl.Trainer(accelerator=accelerator,
                         devices=args.devices,
                         max_epochs=args.epochs,
                         gradient_clip_val=args.clip_gradients,
                         limit_train_batches=args.limit_train_batches,
                         limit_val_batches=args.limit_val_batches,
                         logger=logger,
                         profiler=args.profiler,
                         detect_anomaly=args.detect_anomaly,
                         callbacks=callbacks,
                         plugins=plugins)

    trainer.fit(model, datamodule=nudata, ckpt_path=args.resume)

if __name__ == '__main__':
    args = configure()
    train(args)
