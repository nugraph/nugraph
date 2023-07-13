from typing import Callable, Optional

import h5py
from pynuml import io

import torch
from torch_geometric.data import Dataset

class H5Dataset(Dataset):
    def __init__(self,
                 filename: str,
                 samples: list[str],
                 transform: Optional[Callable] = None):
        super().__init__(transform=transform)
        self._interface = io.H5Interface(h5py.File(filename))
        self._samples = samples
        print('WARNING: temporary hack fix for 3D vertex in DataLoader. this must be fixed in pynuml before merging')

    def len(self) -> int:
        return len(self._samples)

    def get(self, idx: int) -> 'pyg.data.HeteroData':
        data = self._interface.load_heterodata(self._samples[idx])
        data['evt'].y_vtx.unsqueeze_(0)
        return data