#!/usr/bin/env python
import argparse
import pynuml

def configure():
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--infile", type=str, required=True,
                      help="input HDF5 file")
    args.add_argument("-o", "--outfile", type=str, required=True,
                      help="output file name")
    args.add_argument('--filter-hits', action='store_true',
                      help='filter out background hits and spacepoints')
    return args.parse_args()  

def process(args):

    f = pynuml.io.File(args.infile)
    processor = pynuml.process.HitGraphProducer(
            file=f,
            labeller=pynuml.labels.SimpleLabels(),
            filter_hits=args.filter_hits)
    out = pynuml.io.H5Out(args.outfile)
    f.process(processor, out)

if __name__ == "__main__":
    args = configure()
    process(args)