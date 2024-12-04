"""Transform to add hierarchical interaction edges"""
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

class TylerHack(BaseTransform):
    """Add simple edge indices to interaction level"""

    def __call__(self, data: HeteroData) -> HeteroData:
        data["sp"].x = torch.empty(data["sp"].num_nodes, 0)
        data["hit"].x = data["hit"].x[:, (0,1,2,3,4,5,6,8)]
        return data
