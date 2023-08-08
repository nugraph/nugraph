#!/usr/bin/env python

import sys
import os
import time
import argparse
import torch
import pytorch_lightning as pl
import nugraph as ng
import h5py
import pynuml
import numpy as np

Data = ng.data.H5DataModule
Model = ng.models.NuGraph2

def configure():
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file for trained model')
    parser.add_argument('--device', type=str, default='gpu',
                        help='Device to use for Model')
    parser.add_argument('--outfile', type=str, required=True,
                        help='Output file name (full path)')
    parser = Data.add_data_args(parser)
    return parser.parse_args()

def test(args):

    # Load dataset
    print('data path =',args.data_path)
    nudata = Data(args.data_path, batch_size=args.batch_size)

    print('using checkpoint =',args.checkpoint)
    model = Model.load_from_checkpoint(args.checkpoint, map_location=torch.device(args.device))

    print('output file =',args.outfile)
    if os.path.isfile(args.outfile):
        print('file exists. Exiting.')
        exit(1)

    accelerator, devices = ng.util.configure_device()
    trainer = pl.Trainer(accelerator=accelerator, devices=devices,
                         #limit_predict_batches=2,
                         logger=None)
    plot = pynuml.plot.GraphPlot(planes=nudata.planes,
                                 classes=nudata.semantic_classes)

    start = time.time()
    out = trainer.predict(model, dataloaders=nudata.test_dataloader())
    end = time.time()
    itime = end - start
    ngraphs = len(nudata.test_dataset)
    print(f'inference for {ngraphs} events is {itime} s (that\'s {itime/ngraphs} s/graph')

    for ib,batch in enumerate(out):
        if ib%10==0: print(ib,'/',len(out))
        md = batch['metadata']
        df = plot.to_dataframe(batch)
        r = np.zeros((len(df),), dtype=int)
        s = np.zeros((len(df),), dtype=int)
        e = np.zeros((len(df),), dtype=int)
        offset = 0
        for p in nudata.planes:
            for ie in range(0,len(md['run'])):
                r[offset+batch[p]['ptr'][ie]:offset+batch[p]['ptr'][ie+1]] = md['run'][ie]
                s[offset+batch[p]['ptr'][ie]:offset+batch[p]['ptr'][ie+1]] = md['subrun'][ie]
                e[offset+batch[p]['ptr'][ie]:offset+batch[p]['ptr'][ie+1]] = md['event'][ie]
            offset = offset + len(batch[p]['id'])
        df['run'] = r
        df['subrun'] = s
        df['event'] = e
        df.to_hdf(args.outfile, 'hits', append=True, format='table') # requires adding pytables to conda env


if __name__ == '__main__':
    args = configure()
    test(args)
