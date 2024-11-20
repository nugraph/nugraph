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
        self.file = h5py.File(filename)
        self.samples = samples

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> NuGraphData:
        key = f"/dataset/{self.samples[idx]}"
        return NuGraphData.load(self.file[key])
