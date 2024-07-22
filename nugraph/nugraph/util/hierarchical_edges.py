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

        # fix optical hierarchical edges, if necessary
        key = ("ophits", "sumpe", "opflashsumpe")
        if key in data.edge_types:
            mask = data[key].edge_index[1] > -1
            data[key].edge_index = data[key].edge_index[:, mask]
            lo, hi = data[key].edge_index
            data["opflashsumpe", "sumpe", "ophits"].edge_index = torch.stack((hi, lo), dim=0)
            lo, hi = data["opflashsumpe", "flash", "opflash"].edge_index
            data["opflash", "flash", "opflashsumpe"].edge_index = torch.stack((hi, lo), dim=0)
            lo, hi = data["opflash", "in", "evt"].edge_index
            data["evt", "in", "opflash"].edge_index = torch.stack((hi, lo), dim=0)

        return data
