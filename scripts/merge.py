#!/usr/bin/env python

import sys
import argparse
import glob
import tqdm
import h5py

def configure():
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('-i', '--input-dir', type=str, required=True,
                        help='Directory containing input HDF5 files')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='Output HDF5 files')
    return parser.parse_args()

def merge(args):

    with h5py.File(args.output_file, 'w') as fout:
        fout.create_group('dataset')
        for fname in tqdm.tqdm(glob.glob(f'{args.input_dir}/*.h5')):
            with h5py.File(fname) as f:
                if f.get('dataset') is None: continue
                for key in f['dataset'].keys():
                    f.copy(f[f'dataset/{key}'], fout['dataset'], key)

if __name__ == '__main__':
    args = configure()
    merge(args)
