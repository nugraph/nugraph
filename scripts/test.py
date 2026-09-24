#!/usr/bin/env python

import os
import time
import argparse

import pandas as pd
import pytorch_lightning as pl
import nugraph as ng
import pynuml
import tqdm


Data = ng.data.H5DataModule
Model = ng.models.NuGraph3


def configure():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Index of GPU device to use",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint file for the trained model",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Output HDF5 file name",
    )
    parser.add_argument(
        "--split",
        choices=("test", "validation"),
        default="test",
        help="Dataset split to export",
    )

    parser = Data.add_data_args(parser)
    return parser.parse_args()


def test(args):
    print("using checkpoint =", args.checkpoint)
    model = Model.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        michel_pos_weight=1.0,
        strict=False,
    )

    expected_features = int(model.hparams.in_features)
    print("checkpoint expects in_features =", expected_features)
    print("data path =", args.data_path)

    # IMPORTANT:
    # Pass model=Model so H5DataModule applies Model.transform(planes),
    # exactly as the training script does.
    nudata = Data(
        args.data_path,
        model=Model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.split == "validation":
        dataloader = nudata.val_dataloader()
        dataset = nudata.val_dataset
    else:
        dataloader = nudata.test_dataloader()
        dataset = nudata.test_dataset

    # Fail loudly if preprocessing still does not match the checkpoint.
    check_batch = next(iter(dataloader))
    actual_features = int(check_batch["hit"].x.shape[-1])

    print("features after NuGraph transform =", actual_features)

    if actual_features != expected_features:
        raise RuntimeError(
            "Input preprocessing still does not match the checkpoint: "
            f"transformed data has {actual_features} hit features, "
            f"but the checkpoint expects {expected_features}. "
            "Do not pad or crop the feature tensor."
        )

    print("output file =", args.outfile)

    if os.path.isfile(args.outfile):
        raise FileExistsError(
            f"Output file already exists: {args.outfile}"
        )

    accelerator, devices = ng.util.configure_device(args.device)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
    )

    plot = pynuml.plot.GraphPlot(
        planes=nudata.planes,
        classes=nudata.semantic_classes,
    )

    start = time.time()

    out = trainer.predict(
        model,
        dataloaders=dataloader,
    )

    elapsed = time.time() - start
    ngraphs = len(dataset)

    print(
        f"inference for {ngraphs} {args.split} events is {elapsed:.3f} s "
        f"({elapsed / ngraphs:.6f} s/graph)"
    )

    frames = []

    for batch in tqdm.tqdm(out):
        for data in batch.to_data_list():
            frames.append(plot.to_dataframe(data))

    if not frames:
        raise RuntimeError("No prediction batches were returned.")

    df = pd.concat(frames, ignore_index=True)
    df.to_hdf(
        args.outfile,
        key="hits",
        format="table",
        mode="w",
    )

    print("saved =", args.outfile)
    print("rows =", len(df))


if __name__ == "__main__":
    test(configure())
