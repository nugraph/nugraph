#!/usr/bin/env python

import sys
import argparse

from nugraph.data import H5DataModule

def configure():
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--data-path', type=str, required=True,
                        help='Location of input data file')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Size of each batch of graphs')
    return parser.parse_args()

def prepare(args):
    H5DataModule.generate_samples(args.data_path)
    H5DataModule.generate_norm(args.data_path, args.batch_size)

if __name__ == '__main__':
    args = configure()
    prepare(args)