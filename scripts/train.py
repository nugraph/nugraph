#!/usr/bin/env python

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None,
                        help='Training instance name, for logging purposes')
    parser.add_argument('--version', type=str, default=None,
                        help='Training version name, for logging purposes')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Output directory to write logs to')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint file to resume training from')
    parser.add_argument('--profiler', type=str, default=None,
                        help='Enable requested profiler')
    parser = Data.add_data_args(parser)
    parser = Model.add_model_args(parser)
    parser = Model.add_train_args(parser)
    args = parser.parse_args()
    # invalid checkpoint path -> None
    if args.resume is not None and not os.path.exists(args.resume): 
        args.resume = None
    return args

def train(args):

    torch.manual_seed(1)

    # Load dataset
    nudata = Data(args.data_path, batch_size=args.batch_size, 
                  shuffle=args.shuffle, balance_frac=args.balance_frac)

    if args.name is not None and args.logdir is not None and args.resume is None: 
        # No checkpoint directory provided
        model = Model(in_features=4,
                      planar_features=args.planar_feats,
                      nexus_features=args.nexus_feats,
                      vertex_aggr=args.vertex_aggr,
                      vertex_lstm_features=args.vertex_lstm_feats,
                      vertex_mlp_features=args.vertex_mlp_feats,
                      planes=nudata.planes,
                      semantic_classes=nudata.semantic_classes,
                      event_classes=nudata.event_classes,
                      num_iters=5,
                      event_head=args.event,
                      semantic_head=args.semantic,
                      filter_head=args.filter,
                      vertex_head=args.vertex,
                      checkpoint=not args.no_checkpointing,
                      lr=args.learning_rate)
        name, logdir, version = args.name, args.logdir, args.version
        os.makedirs(os.path.join(logdir, args.name), exist_ok=True) # Make a checkpoint directory
        ckpt_path = None # No initial weights for the model to load

    elif args.resume is not None: 
        # If checkpoint directory is provided
        files = os.listdir(args.resume)
        files.sort(key=lambda x: os.path.getctime(os.path.join(args.resume, x)))
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(args.resume, x)))
        ckpt_path = os.path.join(args.resume, latest_file) # Find most recent filepath
        model = Model.load_from_checkpoint(ckpt_path)
        name, logdir, version = args.name, args.logdir, args.version
    else:
        raise Exception('You must pass either the --name and --logdir arguments to start an existing training, or the --resume argument to resume an existing one.')

    logger = pl.loggers.TensorBoardLogger(save_dir=logdir,
                                          name=name, version=version,
                                          default_hp_metric=False)

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
    ]

    plugins = [
        SLURMEnvironment(requeue_signal=signal.SIGUSR1),
    ]

    accelerator, devices = ng.util.configure_device()
    trainer = pl.Trainer(accelerator=accelerator, devices=devices,
                         max_epochs=args.epochs,
                         limit_train_batches=args.limit_train_batches,
                         limit_val_batches=args.limit_val_batches,
                         logger=logger, profiler=args.profiler,
                         callbacks=callbacks, plugins=plugins)

    trainer.fit(model, datamodule=nudata, ckpt_path=ckpt_path)
    trainer.test(datamodule=nudata)

if __name__ == '__main__':
    args = configure()
    train(args)
