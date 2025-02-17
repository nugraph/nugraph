"""NuGraph data object"""
import h5py
import numpy as np
import torch
from torch_geometric.data import HeteroData

N_IT = "particle-truth" # true instance node store
N_IP = "particle" # predicted instance node store
E_H_IT = ("hit", "cluster-truth", N_IT) # hit to true instance edges
E_H_IP = ("hit", "cluster", N_IP) # hit to predicted instance edges

class NuGraphData(HeteroData):
    """NuGraph data object"""

    # pylint: disable=abstract-method

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def y_i(self) -> torch.Tensor:
        """Return true instance labels on hits"""
        y_i = torch.empty_like(self["hit"].y_semantic).fill_(-1)
        i, j = self[E_H_IT].edge_index
        y_i[i] = j
        return y_i

    def x_i(self) -> torch.Tensor:
        """Return predicted instance labels on hits"""
        x_i = torch.empty_like(self["hit"].y_semantic).fill_(-1)
        i, j = self[E_H_IP].edge_index
        x_i[i] = j
        return x_i

    def __inc__(self, key: str, value: torch.Tensor, *args, **kwargs) -> int:
        """Increment tensor values"""
        if key == "x_instance":
            return self[N_IP].num_nodes
        if key == "y_instance":
            return self[N_IT].num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def save(self, file: h5py.File, name: str) -> None:
        """Save NuGraph data object to HDF5 file

        Args:
            file: HDF5 file to save graph to
            name: Name of dataset to save graph to
        """

        data = []
        fields = []

        nodes, edges = self.metadata()
        stores = [(key, key) for key in nodes] + [(key, "_".join(key)) for key in edges]

        # loop over data tensors
        for store, prefix in stores:

            # check for syntax errors in data store names
            if isinstance(store, tuple) and prefix.count("_") != 2:
                raise ValueError((f"\"{prefix}\" is not a valid edge store name!"
                                  " Too many underscores."))
            if isinstance(store, str) and prefix.count("_"):
                raise ValueError((f"\"{prefix}\" is not a valid node store name!"
                                  " Underscores are not supported."))

            # add attributes to compound data object
            for attr in self[store].keys():
                key = f"{prefix}/{attr}"
                val = self[store][attr]
                if np.isscalar(val):
                    data.append(val)
                    fields.append((key, type(val)))
                elif val.nelement() == 0: # save tensor with zero-sized dimension as a scalar 0
                    # HDF5 compound data type does not allow zero-size dimension
                    # ValueError: Zero-sized dimension specified (zero-sized dimension specified)
                    data.append(0)
                    fields.append((key, val.numpy().dtype))
                else:
                    val = val.numpy() # convert a tensor to numpy
                    data.append(val)
                    fields.append((key, val.dtype, val.shape))

        # create a scalar dataset of compound data type
        ctype = np.dtype(fields)
        ds = file.create_dataset(name, shape=(), dtype=ctype, data=tuple(data))
        del ctype, fields, data, ds

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

        # handle empty node tensors
        for node_type in data.node_types:
            n = data[node_type]
            if n.num_nodes is not None and not hasattr(n, "x"):
                n.x = torch.empty([n.num_nodes, 0])

        return data
