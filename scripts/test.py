#!/usr/bin/env python

import sys
import os
import time
import argparse
import torch
import pytorch_lightning as pl
import nugraph as ng

Data = ng.data.H5DataModule
Model = ng.models.NuGraph2

def configure():
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file for trained model')
    parser.add_argument('--devices', nargs='+', type=int, default=None,
                        help='List of devices to test with')
    parser = Data.add_data_args(parser)
    return parser.parse_args()

def test(args):

    # Load dataset
    nudata = Data(args.data_path,
                  batch_size=args.batch_size,
                  planes=['u','v','y'],
                  classes=['HIP','MIP','shower','michel','diffuse'],)

    model = Model.load_from_checkpoint(args.checkpoint, event_head=False)

    if args.devices is None:
        print('No devices specified – running inference on CPU')

    accelerator = 'cpu' if args.devices is None else 'gpu'
    trainer = pl.Trainer(accelerator=accelerator,
                         devices=args.devices,
                         logger=None)
    start = time.time()
    trainer.test(model, datamodule=nudata)
    end = time.time()
    itime = end - start
    ngraphs = len(nudata.test_dataset)
    print(f'inference for {ngraphs} events is {itime} s (that\'s {itime/ngraphs} s/graph')

if __name__ == '__main__':
    args = configure()
    test(args)
