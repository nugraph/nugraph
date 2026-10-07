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
    args.add_argument("--ancestry", action="store_true",
                      help="add particle parents, processes and start/end positions")
    args.add_argument("--corrected-positions", action="store_true",
                      help="use space-charge-corrected particle positions (MicroBooNE)")
    args.add_argument("--split-delta-rays", action="store_true",
                      help="give delta rays their own instances instead of their parent's")
    return args.parse_args()

def process(args):

    # open input file
    f = pynuml.io.File(args.infile)

    # create graph processor
    processor = pynuml.process.HitGraphProducer(
            file=f,
            semantic_labeller=pynuml.labels.StandardLabels(split_delta_rays=args.split_delta_rays),
            event_labeller=pynuml.labels.FlavorLabels(),
            label_vertex=args.label_vertex,
            label_position=args.label_position,
            optical=args.optical,
            ancestry=args.ancestry,
            corrected_positions=args.corrected_positions)

    # create output file stream
    out = pynuml.io.H5Out(args.outfile)

    # run processing
    f.process(processor, out)

if __name__ == "__main__":
    args = configure()
    process(args)
