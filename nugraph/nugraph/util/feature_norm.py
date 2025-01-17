"""Feature normalization transform"""
import torch
from torch_geometric.transforms import BaseTransform
from torchmetrics import Metric

class FeatureNormMetric(Metric):
    """
    Metric for calculating feature tensor normalizations

    Args:
        num_features: number of features in feature tensor
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.add_state('n', default=torch.zeros(num_features))
        self.add_state('mean', default=torch.zeros(num_features))
        self.add_state('std', default=torch.zeros(num_features))

    def update(self, x: torch.Tensor): # pylint: disable=arguments-differ

        assert x.dim() == 2

        # pylint: disable=access-member-before-definition
        n1, m1, s1 = self.n, self.mean, self.std
        n2 = x.shape[0]
        m2 = x.mean(dim=0)
        s2 = x.std(dim=0)

        # pylint: disable=attribute-defined-outside-init
        self.n = n1 + n2
        self.mean = ((n1*m1)+(n2*m2)) / self.n
        d1 = s1.square() + (self.mean - m1).square()
        d2 = s2.square() + (self.mean - m2).square()
        self.std = (((n1*d1) + (n2*d2)) / self.n).sqrt()

    def compute(self):
        return torch.stack((self.mean,self.std), dim=0)

class FeatureNorm(BaseTransform):
    """
    Transform to normalize 2D graph node features.
    
    Args:
        norm: Tensor of hit feature tensor normalization values, or dictionary
              of planar normalization tensors for first generation inputs
    """
    def __init__(self, norm: torch.Tensor | dict[str, torch.Tensor]):
        super(FeatureNorm, self).__init__()
        self.norm = norm

    def __call__(self, data: "pyg.data.HeteroData") -> "pyg.data.HeteroData":

        # feature norm for first-gen graphs
        if isinstance(self.norm, dict):
            for p, (mean, std) in self.norm.items():
                data[p].x = (data[p].x - mean[None,:]) / std[None,:]

        # feature norm for second-gen graphs
        else:
            mean, std = self.norm
            data["hit"].x = (data["hit"].x - mean[None,:]) / std[None,:]

        return data
