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
        data["interaction"].num_nodes = 1
        for p in self.planes + ["sp"]:
            edge_index = torch.zeros(2, data[p].num_nodes, dtype=torch.long)
            edge_index[0] = torch.arange(data[p].num_nodes)
            data[p, "in", "interaction"].edge_index = edge_index
        return data
