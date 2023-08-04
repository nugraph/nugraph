#!/usr/bin/env python

import os
import argparse
import pytorch_lightning as pl
import pynuml
import torch
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
    parser.add_argument('--device', type=str, default='gpu',
                        help='Device to use for Model')
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
        model = Model.load_from_checkpoint(args.checkpoint, map_location=torch.device(args.device))
        model.freeze()

        accelerator, devices = ng.util.configure_device()
        trainer = pl.Trainer(accelerator=accelerator, devices=devices,
                             limit_predict_batches=args.limit_predict_batches,
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
            #somehow the x_semantic table is lost when iterating over the batch, so need to add it back
            offsets = {'u': batch['u']['ptr'], 'v': batch['v']['ptr'], 'y': batch['y']['ptr']}
            x_semantics = {'u': batch['u']['x_semantic'], 'v': batch['v']['x_semantic'], 'y': batch['y']['x_semantic']}
            for ie,data in enumerate(batch.to_data_list()):
                for p in ['u','v','y']:
                    data[p]['x_semantic'] = x_semantics[p][offsets[p][ie]:offsets[p][ie+1]]
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
