"""NuGraph dataset"""
from typing import Callable, Optional
import h5py
from torch_geometric.data import Dataset

from pynuml.data import NuGraphData

class NuGraphDataset(Dataset):
    """NuGraph dataset

    Args:
        filename: Name of dataset file
        samples: List of graph object dataset names in file
        transform: Transforms to apply to graph objects
    """
    def __init__(self,
                 filename: str,
                 samples: list[str],
                 transform: Optional[Callable] = None):
        super().__init__(transform=transform)
        self.filename = filename
        self._file = None
        self.samples = samples

    @property
    def file(self) -> h5py.File:
        """HDF5 file, opened on first access in each process"""
        if self._file is None:
            self._file = h5py.File(self.filename)
        return self._file

    def __getstate__(self) -> dict:
        # h5py file handles cannot be pickled, so DataLoader workers started
        # with spawn or forkserver open their own handle instead
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> NuGraphData:
        key = f"/dataset/{self.samples[idx]}"
        return NuGraphData.load(self.file[key])
