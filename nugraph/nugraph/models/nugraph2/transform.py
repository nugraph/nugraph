"""NuGraph2 data transform"""
import torch
from torch_geometric.transforms import BaseTransform

from pynuml.data import NuGraphData

class Transform(BaseTransform):
    """NuGraph2 data transform"""

    def __call__(self, data: NuGraphData) -> NuGraphData:
        """
        Apply transform for compatibility with NuGraph2 model

        Args:
           data: NuGraph data object to transform
        """

        if "hit" in data.node_types:
            # where are planes stored? can we find them in the data object?

        n = data["hit"]
        n.x = torch.cat(((n.plane-1).unsqueeze(dim=1), n.x), dim=-1)
        return data
