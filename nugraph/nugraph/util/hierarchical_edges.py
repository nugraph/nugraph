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
        data["hit", "plane", "hit"].edge_index = torch.cat(edge_plane, dim=1)
        data["hit", "nexus", "sp"].edge_index = torch.cat(edge_nexus, dim=1)

        # add plane index to feature tensor
        for i, p in enumerate(self.planes):
            ip = torch.empty_like(data[p].x[:,0]).fill_(i).unsqueeze(1)
            data[p].x = torch.cat([data[p].x, ip], dim=1)
        
        # merge planar node stores
        for attr in data[self.planes[0]].node_attrs():
            data["hit"][attr] = torch.cat([data[p][attr] for p in self.planes], dim=0)
        for p in self.planes:
            del data[p]

        # add edges to and from event node
        data["evt"].num_nodes = 1
        for p in self.planes + ["sp"]:
            lo = torch.arange(data[p].num_nodes, dtype=torch.long)
            hi = torch.zeros(data[p].num_nodes, dtype=torch.long)
            data[p, "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)
            data["evt", "owns", p].edge_index = torch.stack((hi, lo), dim=0)

        # add edges from nexus to plane
        for p in self.planes:
            lo, hi = data[p, "nexus", "sp"].edge_index
            data["sp", "nexus", p].edge_index = torch.stack((hi, lo), dim=0)

        return data
