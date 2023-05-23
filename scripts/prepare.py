#!/usr/bin/env python

import sys
import argparse
import nugraph

Data = nugraph.data.H5DataModule

def configure():
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--data-path', type=str, required=True,
                        help='Location of input data file')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Size of each batch of graphs')
    parser.add_argument('--num-workers', type=int, default=5,
                        help='Number of worker processes for data loading')
    return parser.parse_args()

def prepare(args):
    nudata = Data(data_path=args.data_path,
                  batch_size=args.batch_size,
                  planes=['u','v','y'],
                  classes=['MIP','HIP','shower','michel','diffuse'],
                  prepare=True)

if __name__ == '__main__':
    args = configure()
    prepare(args)
