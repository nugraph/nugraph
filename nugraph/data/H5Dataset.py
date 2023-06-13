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

    def len(self) -> int:
        return len(self._samples)

    def get(self, idx: int) -> 'pyg.data.HeteroData':
        data = self._interface.load_heterodata(self._samples[idx])
        for p in data.collect('x'):
            y_filter = data[p].y_f

            y_semantic = torch.empty(y_filter.size()).long()
            y_semantic[y_filter] = data[p].y_s
            y_semantic[~y_filter] = -1
            data[p].y_semantic = y_semantic

            y_instance = torch.empty(y_filter.size()).long()
            y_instance[y_filter] = data[p].y_i
            y_instance[~y_filter] = -1
            data[p].y_instance = y_instance

            del data[p].y_f
            del data[p].y_s
            del data[p].y_i

        return data