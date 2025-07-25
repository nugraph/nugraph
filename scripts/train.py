#!/usr/bin/env python

import os
import argparse
import pathlib
import signal
import warnings

import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import nugraph as ng

torch.set_num_threads(4)

warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')

Data = ng.data.H5DataModule
Model = ng.models.NuGraph3

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=None,
                        help="Index of GPU device to train with")
    parser.add_argument('--logger', type=str, default="wandb",
                        choices=("wandb", "tensorboard"),
                        help="Which logging method to use")
    parser.add_argument('--name', type=str, default=None,
                        help='Training instance name, for logging purposes')
    parser.add_argument('--version', type=str, default=None,
                        help='Training version name, for logging purposes')
    parser.add_argument("--project", type=str, default="nugraph3",
                        help="wandb project to log to")
    parser.add_argument("--resume", type=str,
                        help="model checkpoint file to resume training with")
    parser.add_argument("--offline", action="store_true",
                        help="write wandb logs offline")
    parser.add_argument('--profiler', type=str, default=None,
                        help='Enable requested profiler')
    parser = Data.add_data_args(parser)
    parser = Model.add_model_args(parser)
    return parser.parse_args()

def train(args):

    torch.manual_seed(1)

    # Load dataset
    nudata = Data(args.data_path, batch_size=args.batch_size,
                  model=Model, shuffle=args.shuffle,
                  balance_frac=args.balance_frac)

    if args.resume:
        model = Model.load_from_checkpoint(args.resume)
    else:
        model = Model.from_args(args, nudata)

    # Configure logger
    if args.logger == "wandb":
        logdir = pathlib.Path(os.environ["NUGRAPH_LOG"])/args.name
        logdir.mkdir(parents=True, exist_ok=True)
        log_model = False if args.offline else "all"
        logger = pl.loggers.WandbLogger(save_dir=logdir, project=args.project,
                                        name=args.name, version=args.version,
                                        log_model=log_model, offline=args.offline)
    elif args.logger == "tensorboard":
        logdir = os.environ["NUGRAPH_LOG"]
        logger = pl.loggers.TensorBoardLogger(save_dir=logdir, name=args.name,
                                              version=args.version, default_hp_metric=False)
    else:
        raise RuntimeError(f"Logger option \"{args.logger}\" not recognized!")

    # configure callbacks
    callbacks = []
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    if isinstance(logger, pl.loggers.WandbLogger) and not args.offline:
        callbacks.append(ModelCheckpoint(monitor="loss/val", mode="min"))

    # configure plugins
    plugins = [
        SLURMEnvironment(requeue_signal=signal.SIGUSR1),
    ]

    model = Model.from_args(args, nudata)

    accelerator, devices = ng.util.configure_device(args.device)
    trainer = pl.Trainer(accelerator=accelerator, devices=devices,
                         max_epochs=args.epochs,
                         limit_train_batches=args.limit_train_batches,
                         limit_val_batches=args.limit_val_batches,
                         logger=logger, profiler=args.profiler,
                         callbacks=callbacks, plugins=plugins)

    trainer.fit(model, datamodule=nudata, ckpt_path=args.resume)

if __name__ == '__main__':
    args = configure()
    train(args)
