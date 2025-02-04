"""NuGraph dataset"""
from typing import Callable, Optional
import h5py
from torch_geometric.data import Dataset

from pynuml.data import NuGraphData

class NuGraphDataset(Dataset): # pylint: disable=abstract-method
    """NuGraph dataset

    Args:
        file: Input HDF5 file
        samples: List of graph object dataset names in file
        transform: Transforms to apply to graph objects
    """
    def __init__(self,
                 file: h5py.File,
                 samples: list[str],
                 transform: Optional[Callable] = None):
        super().__init__(transform=transform)
        self.file = file
        self.samples = samples

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> NuGraphData:
        key = f"/dataset/{self.samples[idx]}"
        return NuGraphData.load(self.file[key])
