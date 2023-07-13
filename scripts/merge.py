#!/usr/bin/env python
import argparse
import os
import glob
import tqdm
import h5py

from nugraph.data import H5DataModule

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='HDF5 file pattern')
    return parser.parse_args()

def merge(args):

    # open final output file
    with h5py.File(args.file, 'w', libver='latest') as fout:
        data_out = fout.create_group('dataset')

        # loop over each input file to merge it in
        for fname in tqdm.tqdm(glob.glob(f'{args.file}*.h5')):
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

            # delete temporary file once it's been merged
            os.remove(fname)

    # prepare dataset
    H5DataModule.generate_samples(args.file)
    H5DataModule.generate_norm(args.file, 64)

if __name__ == '__main__':
    args = configure()
    merge(args)