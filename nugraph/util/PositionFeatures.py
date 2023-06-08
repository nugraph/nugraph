from torch import cat
from torch_geometric.transforms import BaseTransform

class PositionFeatures(BaseTransform):
    '''Add node position to node feature tensor'''
    def __init__(self, planes: list[str]):
        super().__init__()
        self.planes = planes

    def __call__(self, data: 'pyg.data.HeteroData') -> 'pyg.data.HeteroData':
        for p in self.planes:
            data[p].x = cat((data[p].pos, data[p].x), dim=-1)
        return data