#!/usr/bin/env python
import argparse
import pynuml

def configure():
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--infile", type=str, required=True,
                      help="input HDF5 file")
    args.add_argument("-o", "--outfile", type=str, required=True,
                      help="output HDF5 file pattern")
    args.add_argument('--label-vertex', action='store_true',
                      help='add true vertex label to graphs')
    args.add_argument("--label-position", action="store_true",
                      help="add true 3D hit position to graphs")
    args.add_argument("--optical", action="store_true",
                      help="add optical hierarchy")
    return args.parse_args()  

def process(args):

    # open input file
    f = pynuml.io.File(args.infile)

    # create graph processor
    processor = pynuml.process.HitGraphProducer(
            file=f,
            semantic_labeller=pynuml.labels.StandardLabels(),
            event_labeller=pynuml.labels.FlavorLabels(),
            label_vertex=args.label_vertex,
            label_position=args.label_position,
            optical=args.optical)

    # create output file stream
    out = pynuml.io.H5Out(args.outfile)

    # run processing
    f.process(processor, out)

if __name__ == "__main__":
    args = configure()
    process(args)
