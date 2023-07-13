#!/usr/bin/env python
import argparse
import pynuml

def configure():
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--infile", type=str, required=True,
                      help="input HDF5 file")
    args.add_argument("-o", "--outfile", type=str, required=True,
                      help="output HDF5 file pattern")
    args.add_argument('--filter-hits', action='store_true',
                      help='filter out background hits and spacepoints')
    return args.parse_args()  

def process(args):

    # open input file
    f = pynuml.io.File(args.infile)

    # create graph processor
    processor = pynuml.process.HitGraphProducer(
            file=f,
            semantic_labeller=pynuml.labels.SimpleLabels(),
            event_labeller=pynuml.labels.FlavorLabels(),
            label_vertex=True,
            filter_hits=args.filter_hits)

    # create output file stream
    out = pynuml.io.H5Out(args.outfile)

    # run processing
    f.process(processor, out)

if __name__ == "__main__":
    args = configure()
    process(args)