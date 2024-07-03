import os
import sys
from typing import Any

import h5py
from mpi4py import MPI


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

    def write_metadata(metadata: dict[str, Any]) -> None:
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

        from .h5interface import H5Interface
        self.interface = H5Interface(self.f)
        # print(f"{rank}: creating {self.fname}")
        # sys.stdout.flush()

    def __call__(self, name: str, obj: Any) -> None:
        """
        for key, val in obj:
            # set chunk sizes to val shape, so there is only one chunk per dataset
            # if isinstance(val, torch.Tensor) and val.nelement() == 0 :
            #   print("zero val ",name,"/",key," shape=",val.shape)
            if isinstance(val, torch.Tensor) and val.nelement() > 0 :
                # Note compressed datasets can only be read/written in MPI collective I/O mode in HDF5
                self.f.create_dataset(f"/{name}/{key}", data=val, chunks=val.shape, compression="gzip")
                # The line below is to not enable chunking/compression
                # self.f.create_dataset(f"/{name}/{key}", data=val)
            else:
                # if data is not a tensor or is empty, then disable chunking/compression
                self.f.create_dataset(f"/{name}/{key}", data=val)
        """
        import numpy as np
        import torch_geometric as pyg

        # collect and construct fields of compound data type
        fields = []
        data = ()

        # special treatment for heterograph object
        if isinstance(obj, pyg.data.HeteroData):
            self.interface.save(name, obj)
            return
        for key, val in obj:
            if np.isscalar(val): # only n_sp is a scalar
                data = data + (val,)
                field = (key, type(val))
            else:
                if val.nelement() == 0: # save tensor with zero-sized dimension as a scalar 0
                    # HDF5 compound data type does not allow zero-size dimension
                    # ValueError: Zero-sized dimension specified (zero-sized dimension specified)
                    val = val.numpy()  # convert a tensor to numpy
                    data = data + (0,)
                    field = (key, val.dtype)
                else:
                    val = val.numpy() # convert a tensor to numpy
                    data = data + (val,)
                    field = (key, val.dtype, val.shape)
            fields.append(field)
        ctype = np.dtype(fields)
        # create a scalar dataset of compound data type
        ds = self.f.create_dataset(f"/{name}", shape=(), dtype=ctype, data=data)
        del ctype, fields, data, ds

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        for key, val in metadata.items():
            self.f[key] = val

    def __del__(self):
        if self.f != None: self.f.close()