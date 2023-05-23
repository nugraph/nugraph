import os
import glob
import h5py
from typing import Callable, List, Optional

import torch
import torch_geometric as tg
from torch_geometric.data import Data, Dataset, HeteroData

from pynuml import io
import numpy as np

class H5Dataset(Dataset):
    def __init__(self,
                 filename: str,
                 samples: List[str],
                 transform: Optional[Callable] = None):
        super(H5Dataset, self).__init__(transform=transform)
        self._interface = io.H5Interface(h5py.File(filename))
        self._samples = samples

    def len(self) -> int:
        return len(self._samples)

    def get(self, idx: int) -> HeteroData:
        data = self._interface.load_heterodata(self._samples[idx])
        data['sp'].x = torch.empty(data['sp'].num_nodes, 0)
        del data['sp'].num_nodes
        for p in ['u', 'v', 'y']:
            data[p, 'plane', p].edge_index = data[p].edge_index
            del data[p].edge_index
            data[p, 'nexus', 'sp'].edge_index = data[p, 'forms', 'sp'].edge_index
            del data[p, 'forms', 'sp']
        return data
