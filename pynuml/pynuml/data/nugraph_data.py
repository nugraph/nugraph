"""NuGraph data object"""
import h5py
import torch
from torch_geometric.data import HeteroData

class NuGraphData(HeteroData):
    """NuGraph data object"""

    # pylint: disable=abstract-method

    def __init__(self):
        super().__init__()

    @classmethod
    def load(cls, dset: h5py.Dataset) -> "NuGraphData":
        """
        Load NuGraph data object from file
        
        Args:
            dset: HDF5 dataset to load graph from
        """
        data = NuGraphData()
        group = dset[()]
        for dataset in group.dtype.names:
            store, attr = dataset.split('/')
            if "_" in store:
                store = tuple(store.split("_"))
            if group[dataset].ndim == 0:
                if attr == 'edge_index': # empty edge tensor
                    data[store][attr] = torch.LongTensor([[],[]])
                else: # scalar
                    data[store][attr] = torch.as_tensor(group[dataset][()])
            else: # multi-dimension array
                data[store][attr] = torch.as_tensor(group[dataset][:])
        return data
