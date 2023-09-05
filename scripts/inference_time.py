#!/usr/bin/env python

import sys
import argparse
import tqdm
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import pytorch_lightning as pl
import nugraph as ng

mpl.style.use('seaborn-v0_8')

Data = ng.data.H5DataModule
Model = ng.models.NuGraph2

def configure():
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file to resume training from')
    parser.add_argument('--use-existing', default=False, action='store_true',
                        help='Load values from file instead of generating')
    parser.add_argument('--benchmark-cpu', default=False, action='store_true',
                        help='Benchmark inference time on CPU')

    parser = Data.add_data_args(parser)
    parser = Model.add_model_args(parser)
    return parser.parse_args()

def plot(args):

    model = Model.load_from_checkpoint(args.checkpoint)

    params = {}
    params['batch_size'] = args.batch_size

    nudata = Data(args.data_path, args.batch_size)

    if args.benchmark_cpu:
        nudata = Data(args.data_path, 1)
        trainer = pl.Trainer(accelerator='cpu', logger=False)
        t0 = time.time()
        trainer.test(model, datamodule=nudata)
        print('inference on CPU takes', (time.time()-t0)/len(nudata.test_dataset), 's/evt')

    if not args.use_existing:
        x = []
        y = []
        for i in range(9):
            batch_size = pow(2, i)
            x.append(batch_size)
            nudata = Data(args.data_path, batch_size)
            accelerator, devices = ng.util.configure_device()
            trainer = pl.Trainer(accelerator=accelerator,
                                 devices=devices, logger=False)
            t0 = time.time()
            trainer.test(model, datamodule=nudata)
            y.append((time.time()-t0)/len(nudata.test_dataset))
        torch.save((x, y), 'params.pt')
    x, y = torch.load('params.pt')
    plt.plot(x, y)
    plt.gca().set_xscale('log', base=2)
    ticks = [ pow(2, i) for i in range(9) ]
    plt.xticks(ticks, labels=ticks)
    plt.xlabel('Batch size')
    plt.ylabel('Inference time per graph [s]')
    plt.savefig('inference-time.png')

if __name__ == '__main__':
    args = configure()
    plot(args)
