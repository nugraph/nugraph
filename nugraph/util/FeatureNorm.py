import torch
from torch_geometric.transforms import BaseTransform
from torchmetrics import Metric

class FeatureNormMetric(Metric):
    def __init__(self, num_features: int):
        super().__init__()
        self.add_state('n', default=torch.zeros(num_features))
        self.add_state('mean', default=torch.zeros(num_features))
        self.add_state('std', default=torch.zeros(num_features))

    def update(self, x: torch.Tensor):

        assert x.dim() == 2

        n1, m1, s1 = self.n, self.mean, self.std
        n2 = x.shape[0]
        m2 = x.mean(dim=0)
        s2 = x.std(dim=0)

        self.n = n1 + n2
        self.mean = ((n1*m1)+(n2*m2)) / self.n
        d1 = s1.square() + (self.mean - m1).square()
        d2 = s2.square() + (self.mean - m2).square()
        self.std = (((n1*d1) + (n2*d2)) / self.n).sqrt()

    def compute(self):
        return torch.stack((self.mean,self.std), dim=0)

class FeatureNorm(BaseTransform):
    """Normalise 2D graph node features."""
    def __init__(self, planes: list[str], norm: dict[str, torch.Tensor]):
        super(FeatureNorm, self).__init__()
        self.norm = norm
        self.planes = planes

    def __call__(self, data: "pyg.data.HeteroData") -> "pyg.data.HeteroData":
        for p in self.planes:
            mean, std = self.norm[p]
            data[p].x = (data[p].x - mean[None,:]) / std[None,:]
        return data