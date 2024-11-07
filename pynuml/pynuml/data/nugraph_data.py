"""NuGraph data object"""
import h5py
import numpy as np
from sklearn.cluster import HDBSCAN
import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import unbatch

N_IT = "particle-truth" # true instance node store
N_IP = "particle" # predicted instance node store
E_H_IT = ("hit", "cluster-truth", N_IT) # hit to true instance edges
E_H_IP = ("hit", "cluster", N_IP) # hit to predicted instance edges

class NuGraphData(HeteroData):
    """NuGraph data object"""

    # pylint: disable=abstract-method

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def true_instance_labels(self, regenerate: bool = False) -> torch.Tensor:
        """Return true instance labels on hits

        Args:
            regenerate: Whether to regenerate labels if already materialized
        """

        # materialize clusters if necessary
        if regenerate or not (hasattr(self["hit"], "y_instance") and \
            hasattr(self["hit"], "y_instance_mask")):
            self.materialize_true_particles()

        # return labels
        self["hit"].y_instance[~self["hit"].y_instance_mask] = -1
        return self["hit"].y_instance

    def predicted_instance_labels(self, regenerate: bool = False) -> None:
        """Return predicted instance labels on hits

        Args:
            regenerate: Whether to regenerate labels if already materialized
        """

        # materialize clusters if necessary
        if regenerate or not (hasattr(self["hit"], "x_instance") and \
            hasattr(self["hit"], "x_instance_mask")):
            self.materialize_predicted_particles()

        # fix background labels and return
        self["hit"].x_instance[~self["hit"].x_instance_mask] = -1
        return self["hit"].x_instance

    def materialize_true_particles(self) -> None:
        """Materialize true particle clusters"""

        # assign particle labels and mask
        i, j = self[E_H_IT].edge_index
        self["hit"].y_instance = torch.empty_like(self["hit"].y_semantic).fill_(-1)
        self["hit"].y_instance[i] = j
        self["hit"].y_instance_mask = self["hit"].y_instance > -1

    def materialize_predicted_particles(self) -> None:
        """Materialize predicted particle nodes"""

        def go(x, f):
            mask = f > 0.5
            i = torch.empty_like(mask, dtype=torch.long)
            i[~mask] = -1
            i[mask] = torch.tensor(HDBSCAN().fit(x[mask]).labels_)
            return i

        if not isinstance(self, Batch):
            i = go(self["hit"].ox, self["hit"].x_filter)
            self["hit"].x_instance = i
            self["hit"].x_instance_mask = i > -1
            return

        coords = unbatch(self["hit"].ox, self["hit"].batch, batch_size=self.num_graphs)
        mask = unbatch(self["hit"].x_filter, self["hit"].batch, batch_size=self.num_graphs)
        labels = [go(x, f) for x, f in zip(coords, mask)]

        out = [NuGraphData(hit={"x_instance": l, "x_instance_mask": l > -1},
                           particle={"x": torch.empty(l.max()+1)})
               for l in labels]
        out = Batch.from_data_list(out)

        def copy_attr(d, store, attr):
            # pylint: disable=protected-access
            setattr(self[store], attr, getattr(d[store], attr))
            if store in self._slice_dict:
                self._slice_dict[store][attr] = d._slice_dict[store][attr]
            else:
                self._slice_dict[store] = {attr: d._slice_dict[store][attr]}
            if store in self._inc_dict:
                self._inc_dict[store][attr] = d._inc_dict[store][attr]
            else:
                self._inc_dict[store] = {attr: d._inc_dict[store][attr]}

        copy_attr(out, "hit", "x_instance")
        copy_attr(out, "hit", "x_instance_mask")
        copy_attr(out, "particle", "x")

    @property
    def y_i(self) -> torch.Tensor:
        """True instance labels"""
        return self.true_instance_labels()

    @property
    def x_i(self) -> torch.Tensor:
        """Predicted instance labels"""
        return self.predicted_instance_labels()

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
        return data
