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
Model = ng.models.NuGraph3

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
    return parser.parse_args()

def train(args):

    torch.manual_seed(1)

    # Load dataset
    nudata = Data(args.data_path, batch_size=args.batch_size, 
                  shuffle=args.shuffle, balance_frac=args.balance_frac)

    if args.name is not None and args.logdir is not None and args.resume is None:
        model = Model.from_args(args, nudata)
        name = args.name
        logdir = args.logdir
        version = args.version
        os.makedirs(os.path.join(logdir, args.name), exist_ok=True)
    elif args.resume is not None and args.name is None and args.logdir is None:
        model = Model.load_from_checkpoint(args.resume)
        stub = os.path.dirname(os.path.dirname(args.resume))
        stub, version = os.path.split(stub)
        logdir, name = os.path.split(stub)
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

    trainer.fit(model, datamodule=nudata, ckpt_path=args.resume)
    trainer.test(datamodule=nudata)

if __name__ == '__main__':
    args = configure()
    train(args)