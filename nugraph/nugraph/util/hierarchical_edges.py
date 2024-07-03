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
