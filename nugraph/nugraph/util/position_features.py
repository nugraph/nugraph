"""Position features transform"""
import torch
from torch_geometric.transforms import BaseTransform

class PositionFeatures(BaseTransform):
    """Append node position tensor to node feature tensor"""

    def __call__(self, data: 'pyg.data.HeteroData') -> 'pyg.data.HeteroData':
        h = data["hit"]
        h.x = torch.cat((h.pos, h.x), dim=-1)
        return data
