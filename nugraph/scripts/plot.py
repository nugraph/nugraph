#!/usr/bin/env python

import argparse
import pynuml
import nugraph as ng
import tqdm

Data = ng.data.H5DataModule
Model = ng.models.NuGraph3

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
    parser.add_argument('--html', action='store_true', default=False,
                        help='Write HTML files')
    parser.add_argument('--png', action='store_true', default=False,
                        help='Write PNG files')
    parser.add_argument('--pdf', action='store_true', default=False,
                        help='Write PDF files')
    parser.add_argument('--limit', type=int, default=100,
                        help='Number of graphs to plot')
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

    nudata = Data(args.data_path,
                  batch_size=args.batch_size)

    if args.checkpoint:
        model = Model.load_from_checkpoint(args.checkpoint, map_location='cpu')
        model.freeze()

    plot = pynuml.plot.GraphPlot(planes=nudata.planes,
                                 classes=nudata.semantic_classes)

    for i in tqdm.tqdm(range(args.limit)):
        data = nudata.test_dataset[i]
        if args.checkpoint:
            model.step(data)
        if args.semantic:
            if args.semantic:
                save(plot, data, args.outdir, 'semantic', 'true', 'true',
                     args.html, args.png, args.pdf)
                if args.checkpoint:
                    save(plot, data, args.outdir, 'semantic', 'pred', 'true',
                         args.html, args.png, args.pdf)
            if args.instance:
                save(plot, data, args.outdir, 'instance', 'true', 'true',
                     args.html, args.png, args.pdf)
                if args.checkpoint:
                    save(plot, data, args.outdir, 'instance', 'pred', 'true',
                         args.html, args.png, args.pdf)
            if args.filter:
                save(plot, data, args.outdir, 'filter', 'true', 'none',
                     args.html, args.png, args.pdf)
                if args.checkpoint:
                    save(plot, data, args.outdir, 'instance', 'pred', 'true',
                         args.html, args.png, args.pdf)

if __name__ == '__main__':
    args = configure()
    plot(args)