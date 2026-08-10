#!/usr/bin/env python
import argparse

import torch
import nugraph as ng

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file for trained NuGraph2 model')
    parser.add_argument('--outfile', type=str, required=True,
                        help='Output TorchScript file name (full path)')
    return parser.parse_args()

def compile(args):

    print('using checkpoint =', args.checkpoint)
    model = ng.models.NuGraph2.load_from_checkpoint(args.checkpoint, map_location='cpu')

    print('exporting TorchScript module...')
    scripted = model.export()

    print('output file =', args.outfile)
    torch.jit.save(scripted, args.outfile)
    print('done.')

if __name__ == '__main__':
    args = configure()
    compile(args)
