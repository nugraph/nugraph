#!/usr/bin/env python

from nugraph.util import set_device
set_device()

import sys
import os
import argparse
import pytorch_lightning as pl
import pynuml
import nugraph as ng
import time

Data = ng.data.H5DataModule
Model = ng.models.NuGraph2

def configure():
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--semantic', action='store_true', default=False,
                        help='Plot semantic labels')
    parser.add_argument('--instance', action='store_true', default=False,
                        help='Plot instance labels')
    parser.add_argument('--filter', action='store_true', default=False,
                        help='Plot filter labels')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint file to resume training from')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory to write plots to')
    parser.add_argument('--limit-predict-batches', type=int, default=10,
                        help='Number of batches to plot')
    parser = Data.add_data_args(parser)
    return parser.parse_args()

def plot(args):

    # Load dataset
    nudata = Data(args.data_path,
                  batch_size=args.batch_size)

    if args.checkpoint is not None:
        model = Model.load_from_checkpoint(args.checkpoint)
        model.freeze()

        trainer = pl.Trainer(limit_predict_batches=args.limit_predict_batches,
                             logger=None)

    plot = pynuml.plot.GraphPlot(planes=nudata.planes,
                                 classes=nudata.semantic_classes)

    if args.checkpoint is None:
        for i, batch in enumerate(nudata.test_dataloader()):
            if args.limit_predict_batches is not None and i >= args.limit_predict_batches:
                break
            for data in batch.to_data_list():
                md = data['metadata']
                name = f'r{md.run.item()}_sr{md.subrun.item()}_evt{md.event.item()}'
                if args.semantic:
                    plot.plot(data, name=f'{args.outdir}/{name}_semantic_true',
                              target='semantic', how='true', filter='true')
                if args.instance:
                    plot.plot(data, name=f'{args.outdir}/{name}_instance_true',
                              target='instance', how='true', filter='true')
                if args.filter:
                    plot.plot(data, name=f'{args.outdir}/{name}_filter_true',
                              target='filter', how='true')
    else: 
        for batch in trainer.predict(model, nudata.test_dataloader()):
            for data in batch:
                md = data['metadata']
                name = f'r{md.run.item()}_sr{md.subrun.item()}_evt{md.event.item()}'
                if args.semantic:
                    plot.plot(data, name=f'{args.outdir}/{name}_semantic_true',
                              target='semantic', how='true', filter='true')
                    plot.plot(data, name=f'{args.outdir}/{name}_semantic_pred',
                              target='semantic', how='pred', filter='true')
                if args.instance:
                    plot.plot(data, name=f'{args.outdir}/{name}_instance_true',
                              target='instance', how='true', filter='true')
                    plot.plot(data, name=f'{args.outdir}/{name}_instance_pred',
                              target='instance', how='pred', filter='true')
                if args.filter:
                    plot.plot(data, name=f'{args.outdir}/{name}_filter_true',
                              target='filter', how='true')
                    plot.plot(data, name=f'{args.outdir}/{name}_filter_pred',
                              target='filter', how='pred')

if __name__ == '__main__':
    args = configure()
    plot(args)