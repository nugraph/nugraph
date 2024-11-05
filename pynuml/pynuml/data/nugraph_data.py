"""NuGraph data object"""
import h5py
import torch
from torch_scatter import scatter_min
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import bipartite_subgraph, degree

N_IT = "particle-truth" # true instance node store
N_IP = "particle" # predicted instance node store
E_H_IT = ("hit", "cluster-truth", N_IT) # hit to true instance edges
E_H_IP = ("hit", "cluster", N_IP) # hit to predicted instance edges

class NuGraphData(HeteroData):
    """NuGraph data object"""

    # pylint: disable=abstract-method

    def __init__(self):
        super().__init__()

    def true_instance_labels(self, regenerate: bool = False) -> None:
        """Return true instance labels on hits

        Args:
            regenerate: Whether to regenerate labels if already materialized
        """

        # materialize clusters if necessary
        if regenerate or not (hasattr(self["hit"], "cluster_true_label") and \
            hasattr(self["hit"], "cluster_true_mask")):
            self.materialize_true_particles()

        # return labels
        ret = torch.empty_like(self["hit"].cluster_true_mask, dtype=torch.long).fill_(-1)
        ret[self["hit"].cluster_true_mask] = self["hit"].cluster_true_label
        return ret

    def predicted_instance_labels(self, regenerate: bool = False) -> None:
        """Return predicted instance labels on hits

        Args:
            regenerate: Whether to regenerate labels if already materialized
        """

        # materialize clusters if necessary
        if regenerate or not (hasattr(self["hit"], "cluster_pred_label") and \
            hasattr(self["hit"], "cluster_pred_mask")):
            self.materialize_predicted_particles()

        # return labels
        ret = torch.empty_like(self["hit"].cluster_pred_mask, dtype=torch.long).fill_(-1)
        ret[self["hit"].cluster_pred_mask] = self["hit"].cluster_pred_label
        return ret

    def materialize_true_particles(self) -> None:
        """Materialize true particle clusters"""

        # assign particle labels and mask
        i, j = self[E_H_IT].edge_index
        self["hit"].cluster_true_label = j
        m = torch.zeros_like(self["hit"].y_semantic, dtype=torch.bool)
        m[i] = True
        self["hit"].cluster_true_mask = m

    def materialize_predicted_particles(self) -> None:
        """Materialize predicted particle nodes"""

        if isinstance(self, Batch):
            raise RuntimeError("Materializing of clusters must not be performed over batch.")

        if not (hasattr(self["hit"], "cluster_pred_label") \
            and hasattr(self["hit"], "cluster_pred_mask")):
            raise RuntimeError(("Cannot call materialize_predicted_particles() without running "
                                "NuGraph3 inference first!"))

        # mask to find condensation points
        fmask = self["hit"].of > 0.1
        self[N_IP].x = torch.empty(fmask.sum(), 0, device=self.device)
        self[N_IP].ox = self["hit"].ox[fmask]
        fidx = fmask.nonzero().squeeze(1)

        e = self[E_H_IP]
        x_hit = self["hit"].ox

        # initial particle instance nodes
        self[N_IP].x = torch.empty(fidx.size(0), 0, device=self.device)
        self[N_IP].ox = x_hit[fmask]

        # add edges from condensation hits to non-condensation hits
        dist = (x_hit[~fmask, None, :] - x_hit[None, fmask, :]).square().sum(dim=2)
        e.edge_index = (dist < 1).nonzero().transpose(0, 1).detach()
        e.distance = dist[e.edge_index[0], e.edge_index[1]].detach()
        e.edge_index[0] = torch.nonzero(~fmask).squeeze(1)[e.edge_index[0]]

        # prune particle instances with no hits
        deg = degree(e.edge_index[1], num_nodes=self[N_IP].num_nodes)
        dmask = deg >= self.min_degree
        e.edge_index, e.distance = bipartite_subgraph( # pylint: disable=unbalanced-tuple-unpacking
            (torch.ones(self["hit"].num_nodes, dtype=torch.bool, device=self.device), dmask),
            e.edge_index, e.distance, size=(self["hit"].num_nodes, self[N_IP].num_nodes),
            relabel_nodes=True)
        self[N_IP].x = self[N_IP].x[dmask]
        self[N_IP].ox = self[N_IP].ox[dmask]

        #  add edges from particle nodes to condensation hits
        pidx = fidx[dmask]
        dist = (x_hit[fidx, None, :] - x_hit[None, pidx, :]).square().sum(dim=2)
        edge_index = (dist < 1).nonzero().transpose(0, 1).detach()
        if edge_index.size(1):
            distance = dist[edge_index[0], edge_index[1]].detach()
            edge_index[0] = fidx[edge_index[0]]
            e.edge_index = torch.cat((e.edge_index, edge_index), dim=1)
            e.distance = torch.cat((e.distance, distance), dim=0)

        # collapse cluster edges into node labels
        _, instances = scatter_min(e.distance, e.edge_index[0], dim_size=self["hit"].num_nodes)
        m = instances < e.num_edges # nodes without edges are set to e.num_edges
        self["hit"].cluster_pred_label = instances[m]
        self["hit"].cluster_pred_mask = m

    def __inc__(self, key: str, value: torch.Tensor, *args, **kwargs) -> int:
        """Increment tensor values"""
        if key == "cluster_pred_label":
            return self[N_IP].num_nodes
        if key == "cluster_true_label":
            return self[N_IT].num_nodes
        return super().__inc__(key, value, *args, **kwargs)

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
