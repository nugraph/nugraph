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
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file for trained model')
    parser.add_argument('--outfile', type=str, required=True,
                        help='Output file name (full path)')
    parser = Data.add_data_args(parser)
    return parser.parse_args()

def test(args):

    print('data path =',args.data_path)
    nudata = Data(args.data_path, batch_size=args.batch_size)

    print('using checkpoint =',args.checkpoint)
    model = Model.load_from_checkpoint(args.checkpoint, map_location='cpu')

    print('output file =',args.outfile)
    if os.path.isfile(args.outfile):
        raise Exception(f'file {args.outfile} already exists!')

    accelerator, devices = ng.util.configure_device()
    trainer = pl.Trainer(accelerator=accelerator, devices=devices,
                         logger=False)
    plot = pynuml.plot.GraphPlot(planes=nudata.planes,
                                 classes=nudata.semantic_classes)

    start = time.time()
    out = trainer.predict(model, dataloaders=nudata.test_dataloader())
    end = time.time()
    itime = end - start
    ngraphs = len(nudata.test_dataset)
    print(f'inference for {ngraphs} events is {itime} s (that\'s {itime/ngraphs} s/graph')

    df = []
    for ib, batch in enumerate(tqdm.tqdm(out)):
        for data in batch.to_data_list():
            df.append(plot.to_dataframe(data))
    df = pd.concat(df)
    df.to_hdf(args.outfile, 'hits', format='table')

if __name__ == '__main__':
    args = configure()
    test(args)