#!/usr/bin/env python

from nugraph.util import set_device
set_device()

import os
import argparse
import pytorch_lightning as pl
import pynuml
import nugraph as ng
import time

Data = ng.data.H5DataModule
Model = ng.models.NuGraph2

def configure():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--write-html', action='store_true', default=False,
                        help='Write HTML files')
    parser.add_argument('--write-png', action='store_true', default=False,
                        help='Write PNG files')
    parser.add_argument('--write-pdf', action='store_true', default=False,
                        help='Write PDF files')
    parser.add_argument('--limit-predict-batches', type=int, default=10,
                        help='Number of batches to plot')
    parser = Data.add_data_args(parser)
    return parser.parse_args()

def save(plot: pynuml.plot.GraphPlot, data: 'pyg.data.HeteroData',
         outdir: str, target: str, how: str, filter: str,
         html: bool, png: bool, pdf: bool):
    md = data['metadata']
    tag = f'r{md.run.item()}_sr{md.subrun.item()}_evt{md.event.item()}'
    name = f'{outdir}/{tag}_{target}_{how}'
    fig = plot.plot(data, target=target, how=how, filter=filter)
    if html: fig.write_html(f'{name}.html')
    if png:  fig.write_image(f'{name}.png')
    if pdf:  fig.write_image(f'{name}.pdf')

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
                if args.semantic:
                    save(plot, data, args.outdir, 'semantic', 'true', 'true',
                         args.write_html, args.write_png, args.write_pdf)
                if args.instance:
                    save(plot, data, args.outdir, 'instance', 'true', 'true',
                         args.write_html, args.write_png, args.write_pdf)
                if args.filter:
                    save(plot, data, args.outdir, 'filter', 'true', 'none',
                         args.write_html, args.write_png, args.write_pdf)
    else: 
        for batch in trainer.predict(model, nudata.test_dataloader()):
            for data in batch:
                md = data['metadata']
                name = f'r{md.run.item()}_sr{md.subrun.item()}_evt{md.event.item()}'
                if args.semantic:
                    save(plot, data, args.outdir, 'semantic', 'true', 'true',
                         args.write_html, args.write_png, args.write_pdf)
                    save(plot, data, args.outdir, 'semantic', 'pred', 'true',
                         args.write_html, args.write_png, args.write_pdf)
                if args.instance:
                    save(plot, data, args.outdir, 'instance', 'true', 'true',
                         args.write_html, args.write_png, args.write_pdf)
                    save(plot, data, args.outdir, 'instance', 'pred', 'true',
                         args.write_html, args.write_png, args.write_pdf)
                if args.filter:
                    save(plot, data, args.outdir, 'filter', 'true', 'none',
                         args.write_html, args.write_png, args.write_pdf)
                    save(plot, data, args.outdir, 'filter', 'pred', 'none',
                         args.write_html, args.write_png, args.write_pdf)

if __name__ == '__main__':
    args = configure()
    plot(args)