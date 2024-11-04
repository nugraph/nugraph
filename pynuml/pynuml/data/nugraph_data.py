"""NuGraph data object"""
from torch_geometric.data import HeteroData

class NuGraphData(HeteroData):
    """NuGraph data object"""

    # pylint: disable=abstract-method

    def __init__(self):
        super().__init__()

