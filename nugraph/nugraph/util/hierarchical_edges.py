"""Transform to add hierarchical interaction edges"""
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

class HierarchicalEdges(BaseTransform):
    """
    Add simple edge indices to interaction level

    Args:
        planes: List of graph planes
    """

    def __init__(self, planes: list[str]):
        super().__init__()
        self.planes = planes

    def __call__(self, data: HeteroData) -> HeteroData:

        # no-op if the graph data is already structured how we want
        if "hit" in data.node_types:
            return data

        # unify planar edges
        edge_plane = []
        edge_nexus = []
        for i, p in enumerate(self.planes):
            offset = 0
            for j in range(i): # get offset from previous planes
                offset += data[self.planes[j]].num_nodes
            edge_plane.append(data[p, "plane", p].edge_index + offset)
            del data[p, "plane", p]
            edge_nexus.append(data[p, "nexus", "sp"].edge_index)
            edge_nexus[-1][0] += offset # increment only the plane node index
            del data[p, "nexus", "sp"]
        data["hit", "delaunay-planar", "hit"].edge_index = torch.cat(edge_plane, dim=1)
        data["hit", "nexus", "sp"].edge_index = torch.cat(edge_nexus, dim=1)

        # add plane index to feature tensor
        for i, p in enumerate(self.planes):
            data[p].plane = torch.empty_like(data[p].x[:,0], dtype=int).fill_(i)
            data[p].x = torch.cat([data[p].x, data[p].plane.unsqueeze(1)], dim=1)

        # merge planar node stores
        for attr in data[self.planes[0]].node_attrs():
            data["hit"][attr] = torch.cat([data[p][attr] for p in self.planes], dim=0)
        for p in self.planes:
            del data[p]

        # add true instance nodes
        if hasattr(data["hit"], "y_instance"):
            y = data["hit"].y_instance
            mask = y != -1
            y = y[mask]
            instances = y.unique()
            # remap instances
            imax = instances.max() + 1 if instances.size(0) else 0
            if instances.size(0) != imax:
                remap = torch.full((imax,), -1, dtype=torch.long)
                remap[instances] = torch.arange(instances.size(0))
                y = remap[y]
            data["particle-truth"].x = torch.empty(instances.size(0), 0)
            edges = torch.stack((mask.nonzero().squeeze(1), y), dim=0).long()
            data["hit", "cluster-truth", "particle-truth"].edge_index = edges
            del data["hit"].y_instance

        # add edges to and from event node
        data["evt"].x = torch.empty((1, 0))
        lo = torch.arange(data["hit"].num_nodes, dtype=torch.long)
        hi = torch.zeros(data["hit"].num_nodes, dtype=torch.long)
        data["hit", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)
        lo = torch.arange(data["sp"].num_nodes, dtype=torch.long)
        hi = torch.zeros(data["sp"].num_nodes, dtype=torch.long)
        data["sp", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)

        return data
