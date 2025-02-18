"""NuGraph3 data transform"""
from torch_geometric.transforms import BaseTransform
from pynuml.data import NuGraphData

class Transform(BaseTransform):
    """NuGraph3 data transform"""

    def __call__(self, data: NuGraphData) -> NuGraphData:
        """
        Apply transform for compatibility with NuGraph3 model

        Args:
           data: NuGraph data object to transform
        """

        if "c" in data["hit"].keys():
            data["hit"].y_position = data["hit"].c
            del data["hit"].c
        return data
