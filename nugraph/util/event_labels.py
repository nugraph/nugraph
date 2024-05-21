"""Transform to fix event truth labels"""
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

class EventLabels(BaseTransform):
    """
    Ensure event truth labels have the correct dimensions
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data: HeteroData) -> HeteroData:

        if not data["evt"].y.ndim:
            data["evt"].y = data["evt"].y.reshape([1])
        return data
