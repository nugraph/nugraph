"""NuGraph2 data transform"""
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import subgraph, bipartite_subgraph

from pynuml.data import NuGraphData

class Transform(BaseTransform):
    """
    NuGraph2 data transform
    
    Args:
        planes: Tuple of detector plane names
    """
    def __init__(self, planes: tuple[str]):
        super().__init__()
        self.planes = planes

    def __call__(self, data: NuGraphData) -> NuGraphData:
        """
        Apply transform for compatibility with NuGraph2 model

        Args:
           data: NuGraph data object to transform
        """

        # if using newer hit node format, revert to old planar format
        if "hit" in data.node_types:

            h = data["hit"]

            # loop over planes
            for i, pname in enumerate(self.planes):

                # split out hit nodes into planar node stores
                p = data[pname]
                idx = (h.plane == i).nonzero().squeeze(dim=1)
                for attr in h.node_attrs():
                    p[attr] = h[attr][idx]

                # transform planar edges
                edge_index = data["hit", "delaunay-planar", "hit"].edge_index
                edge_index = subgraph(idx, edge_index, relabel_nodes=True)
                data[pname, "plane", pname].edge_index = edge_index

                # transform nexus edges
                edge_index = data["hit", "nexus", "sp"].edge_index
                n_sp = data["sp"].num_nodes
                edge_index = bipartite_subgraph(
                    (idx, torch.arange(n_sp)), edge_index,
                    size=(h.num_nodes, n_sp), relabel_nodes=True)
                data[pname, "nexus", "sp"].edge_index = edge_index

        # ensure event truth labels have correct format
        evt = data["evt"]
        if not evt.y.ndim:
            evt.y = evt.y.reshape([1])

        # concatenate position tensor onto node features
        for pname in self.planes:
            p = data[pname]
            p.x = torch.cat((p.pos, p.x), dim=-1)

        return data
