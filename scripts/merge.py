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
                        help='Output HDF5 file')
    return parser.parse_args()

def merge(args):

    # open single output file
    with h5py.File(args.output_file, 'w', libver='latest') as fout:
        data_out = fout.create_group('dataset')
        # loop over each input file to merge it in
        for fname in tqdm.tqdm(glob.glob(f'{args.input_dir}/*.h5')):
            with h5py.File(fname) as fin:
                # loop over keys in input file
                for key in fin.keys():
                    data_in = fin[key]
                    # if it's the dataset group, loop over graphs and write those
                    if key == 'dataset':
                        for graph in data_in.keys():
                            fin.copy(data_in[graph], data_out, graph)
                    # otherwise it's metadata, so just copy it directly
                    else:
                        fin.copy(data_in, fout, key)

if __name__ == '__main__':
    args = configure()
    merge(args)
