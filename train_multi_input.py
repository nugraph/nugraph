#!/usr/bin/env python

import os
import sys
import glob
import signal
import pathlib
import argparse
import warnings

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset

import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch_geometric.loader import DataLoader

sys.path.append(f"{os.environ['HOME']}/nugraph/pynuml")
sys.path.append(f"{os.environ['HOME']}/nugraph/nugraph")
os.environ["NUGRAPH_DIR"] = f"{os.environ['HOME']}/nugraph"
os.environ["NUGRAPH_LOG"] = f"{os.environ['HOME']}/logs"

import nugraph as ng
from nugraph.data import NuGraphDataset, BalanceSampler

torch.set_num_threads(4)
warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')

Model = ng.models.NuGraph3


def resolve_h5_inputs(items):
    files = []

    for item in items:
        p = pathlib.Path(item)

        if p.is_file():
            if p.suffix.lower() in (".h5", ".hdf5"):
                files.append(str(p.resolve()))
            continue

        if p.is_dir():
            for f in sorted(p.glob("*.h5")):
                files.append(str(f.resolve()))
            for f in sorted(p.glob("*.hdf5")):
                files.append(str(f.resolve()))
            continue

        # glob pattern
        for match in sorted(glob.glob(item)):
            mp = pathlib.Path(match)
            if mp.is_file() and mp.suffix.lower() in (".h5", ".hdf5"):
                files.append(str(mp.resolve()))
            elif mp.is_dir():
                for f in sorted(mp.glob("*.h5")):
                    files.append(str(f.resolve()))
                for f in sorted(mp.glob("*.hdf5")):
                    files.append(str(f.resolve()))

    files = list(dict.fromkeys(files))

    if not files:
        raise FileNotFoundError(
            f"No HDF5 files found from --data-path: {items}"
        )

    return files


def read_h5_info(filename):
    with h5py.File(filename, "r") as f:
        try:
            planes = f["planes"].asstr()[()].tolist()
            semantic_classes = f["semantic_classes"].asstr()[()].tolist()
        except KeyError as e:
            raise RuntimeError(
                f"{filename} is missing required metadata: {e}. "
                "Expected at least 'planes' and 'semantic_classes'."
            )

        gen = f["gen"][()].item() if "gen" in f else 1

        event_classes = (
            f["event_classes"].asstr()[()].tolist()
            if "event_classes" in f else None
        )

        try:
            train_samples = f["samples/train"].asstr()[()]
            val_samples = f["samples/validation"].asstr()[()]
            test_samples = f["samples/test"].asstr()[()]
        except KeyError:
            raise RuntimeError(
                f"{filename} is missing sample splits. "
                "Run H5DataModule.generate_samples(filename) on this file first."
            )

        try:
            train_datasize = f["datasize/train"][()]
        except KeyError:
            raise RuntimeError(
                f"{filename} is missing 'datasize/train'. "
                "Run H5DataModule.generate_samples(filename) on this file first."
            )

    return {
        "planes": planes,
        "semantic_classes": semantic_classes,
        "gen": gen,
        "event_classes": event_classes,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "train_datasize": np.asarray(train_datasize),
    }


class MultiH5DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_paths,
                 model=None,
                 batch_size=64,
                 num_workers=5,
                 shuffle='random',
                 balance_frac=0.1):
        super().__init__()

        warnings.filterwarnings("ignore", ".*does not have many workers.*")

        self.files = resolve_h5_inputs(data_paths)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance_frac = balance_frac

        if shuffle not in ("random", "balance"):
            raise ValueError('shuffle argument must be "random" or "balance".')
        self.shuffle = shuffle

        # read first file as reference
        ref = read_h5_info(self.files[0])
        self.planes = ref["planes"]
        self.semantic_classes = ref["semantic_classes"]
        self.gen = ref["gen"]
        self.event_classes = ref["event_classes"]

        transform = model.transform(self.planes) if model else None

        train_sets = []
        val_sets = []
        test_sets = []
        train_sizes = []

        for fname in self.files:
            info = read_h5_info(fname)

            # check metadata consistency across files
            if info["planes"] != self.planes:
                raise RuntimeError(f"Inconsistent planes in {fname}")
            if info["semantic_classes"] != self.semantic_classes:
                raise RuntimeError(f"Inconsistent semantic_classes in {fname}")
            if info["gen"] != self.gen:
                raise RuntimeError(f"Inconsistent gen in {fname}")
            if info["event_classes"] != self.event_classes:
                raise RuntimeError(f"Inconsistent event_classes in {fname}")

            train_sets.append(
                NuGraphDataset(fname, info["train_samples"], transform)
            )
            val_sets.append(
                NuGraphDataset(fname, info["val_samples"], transform)
            )
            test_sets.append(
                NuGraphDataset(fname, info["test_samples"], transform)
            )

            train_sizes.append(info["train_datasize"])

        self.train_dataset = (
            train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
        )
        self.val_dataset = (
            val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
        )
        self.test_dataset = (
            test_sets[0] if len(test_sets) == 1 else ConcatDataset(test_sets)
        )

        self.train_datasize = np.concatenate(train_sizes)

        print(f"[INFO] Using {len(self.files)} input HDF5 files")
        for f in self.files:
            print(f"  - {f}")
        print(f"[INFO] Total train graphs: {len(self.train_dataset)}")
        print(f"[INFO] Total val graphs  : {len(self.val_dataset)}")
        print(f"[INFO] Total test graphs : {len(self.test_dataset)}")

    def train_dataloader(self):
        if self.shuffle == 'balance':
            shuffle = False
            sampler = BalanceSampler.BalanceSampler(
                datasize=self.train_datasize,
                batch_size=self.batch_size,
                balance_frac=self.balance_frac
            )
        else:
            shuffle = True
            sampler = None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=True,
            shuffle=shuffle,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size
        )


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

    # multi-file data args
    parser.add_argument('--data-path', nargs='+', required=True,
                        help=('One or more input HDF5 files, directories, or '
                              'glob patterns'))
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Size of each batch of graphs')
    parser.add_argument('--num-workers', type=int, default=5,
                        help='Number of data loader worker processes')
    parser.add_argument('--limit_train_batches', type=int, default=None,
                        help='Max number of training batches to be used')
    parser.add_argument('--limit_val_batches', type=int, default=None,
                        help='Max number of validation batches to be used')
    parser.add_argument('--shuffle', type=str, default='balance',
                        choices=('random', 'balance'),
                        help='Dataset shuffling scheme to use')
    parser.add_argument('--balance-frac', type=float, default=0.1,
                        help='Fraction of dataset to use for workload balancing')

    parser = Model.add_model_args(parser)
    return parser.parse_args()


def train(args):
    torch.manual_seed(1)

    nudata = MultiH5DataModule(
        args.data_path,
        batch_size=args.batch_size,
        model=Model,
        shuffle=args.shuffle,
        balance_frac=args.balance_frac,
        num_workers=args.num_workers
    )

    if args.resume:
        model = Model.load_from_checkpoint(args.resume)
    else:
        model = Model.from_args(args, nudata)

    # Configure logger
    if args.logger == "wandb":
        logdir = pathlib.Path(os.environ["NUGRAPH_LOG"]) / args.name
        logdir.mkdir(parents=True, exist_ok=True)
        log_model = False if args.offline else "all"
        logger = pl.loggers.WandbLogger(
            save_dir=logdir,
            project=args.project,
            name=args.name,
            version=args.version,
            log_model=log_model,
            offline=args.offline
        )
    elif args.logger == "tensorboard":
        logdir = os.environ["NUGRAPH_LOG"]
        logger = pl.loggers.TensorBoardLogger(
            save_dir=logdir,
            name=args.name,
            version=args.version,
            default_hp_metric=False
        )
    else:
        raise RuntimeError(f'Logger option "{args.logger}" not recognized!')

    callbacks = []
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    if isinstance(logger, pl.loggers.WandbLogger) and not args.offline:
        callbacks.append(ModelCheckpoint(monitor="loss/val", mode="min"))

    plugins = [
        SLURMEnvironment(requeue_signal=signal.SIGUSR1),
    ]

    accelerator, devices = ng.util.configure_device(args.device)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        logger=logger,
        profiler=args.profiler,
        callbacks=callbacks,
        plugins=plugins,
        accumulate_grad_batches=1,
        precision="16-mixed"
    )

    trainer.fit(model, datamodule=nudata, ckpt_path=args.resume)


if __name__ == '__main__':
    args = configure()
    train(args)
