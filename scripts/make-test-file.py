#!/usr/bin/env python
"""Script to make reduced event HDF5 file for testing"""

import argparse
import h5py
import numpy as np

def configure():
    """Configure arguments for make_test_file.py script"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True,
                        help="Name of input event HDF5 file")
    parser.add_argument("--outfile", type=str, required=True,
                        help="Name of output test file")
    parser.add_argument("--num-evts", "-n", type=int, default=50,
                        help="Number of events in output test file")
    return parser.parse_args()

def main(args): # pylint: disable=too-many-locals
    """Main function for make_test_file.py script"""

    with h5py.File(args.infile) as f:

        slc = {key: slice(0, 0) for key in f}
        data = {}

        # copy datasets
        for i in range(args.num_evts):

            # get event ID
            event_id = f["event_table/event_id"][i]

            # loop over HDF5 groups
            for gk, grp in f.items():

                # add group to dictionary if not present
                if gk not in data:
                    data[gk] = {}

                # sequencing metadata for this event and group
                mask = grp["event_id.seq_cnt"][:,0] == i
                inc = grp["event_id.seq_cnt"][mask,1].item() if mask.sum() else 0
                start = slc[gk].stop
                stop = start + inc
                slc[gk] = slice(start, stop)

                # check the event IDs match
                if not (grp["event_id"][slc[gk]] == event_id).all():
                    raise RuntimeError("Event ID mismatch found while iterating!")

                # loop over datasets in group
                for dk, dset in grp.items():

                    # skip sequencing metadata
                    if ".seq" in dk:
                        continue

                    # add dataset to dictionary if not present
                    if dk not in data[gk]:
                        data[gk][dk] = []

                    # append rows to dictionary
                    data[gk][dk].append(dset[slc[gk]])

        # loop over groups
        for gk, grp in f.items():

            # concatenate datasets
            for dk, dset in grp.items():
                if ".seq" in dk:
                    continue
                data[gk][dk] = np.concatenate(data[gk][dk], axis=0)

            # add metadata
            seq = grp["event_id.seq"][()]
            data[gk]["event_id.seq"] = seq[seq < args.num_evts]

            seq_cnt = grp["event_id.seq_cnt"][()]
            data[gk]["event_id.seq_cnt"] = seq_cnt[seq_cnt[:, 0] < args.num_evts]

    with h5py.File(args.outfile, "w") as f:
        for gk, grp in data.items():
            for dk, dset in grp.items():
                f[f"{gk}/{dk}"] = dset

if __name__ == "__main__":
    main(configure())
