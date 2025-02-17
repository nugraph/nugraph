import os
import sys
from typing import Any

import h5py
from mpi4py import MPI

from ..data import NuGraphData

class PTOut:
    def __init__(self, outdir: str):
        self.outdir = outdir
        isExist = os.path.exists(outdir)
        if not isExist:
            rank = MPI.COMM_WORLD.Get_rank()
            if rank == 0:
                print("Error: output directory does not exist", outdir)
            sys.stdout.flush()
            MPI.COMM_WORLD.Abort(1)

    def __call__(self, name: str, obj: Any) -> None:
        import torch
        torch.save(obj, os.path.join(self.outdir, name)+".pt")

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        raise NotImplementedError

    def exists(self, name: str) -> bool:
        return os.path.exists(os.path.join(self.outdir, name)+".pt")

class H5Out:
    def __init__(self, fname: str, overwrite: bool = False):
        # This implements one-file-per-process I/O strategy.
        # append MPI process rank to the output file name
        rank = MPI.COMM_WORLD.Get_rank()
        file_ext = ".{:04d}.h5"
        self.fname = fname + file_ext.format(rank)
        if os.path.exists(self.fname):
            if overwrite:
                os.remove(self.fname)
            else:
                print(f"Error: file already exists: {self.fname}")
                sys.stdout.flush()
                MPI.COMM_WORLD.Abort(1)
        # open/create the HDF5 file
        self.f = h5py.File(self.fname, "w")

    def __call__(self, name: str, obj: NuGraphData) -> None:
        obj.save(self.f, f"dataset/{name}")

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        for key, val in metadata.items():
            self.f[key] = val

    def __del__(self):
        if self.f is not None:
            self.f.close()
