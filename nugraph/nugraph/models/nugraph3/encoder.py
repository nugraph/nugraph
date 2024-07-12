"""NuGraph3 encoder"""
from torch import nn

from .types import TD

class Encoder(nn.Module):
    """
    NuGraph3 encoder
    
    Args:
        in_features: Number of input node features
        planar_features: Number of planar node features
        planes: Tuple of planes
    """
    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 planes: tuple[str]):
        super().__init__()
        self.net = nn.Linear(in_features, planar_features)
        self.planes = planes

    def forward(self, x: TD) -> TD:
        """
        NuGraph3 encoder forward pass
        
        Args:
            x: Input feature tensor dictionary
        """
        return {p: self.net(x[p]) for p in self.planes}
