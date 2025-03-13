"""Input feature normalization module"""
import torch
from pytorch_lightning import LightningModule

P = torch.nn.Parameter

class InputNorm(LightningModule):
    """
    PyTorch module to normalize input features

    Args:
        num_feats: Number of tensor features
    """
    def __init__(self, num_features: int):
        super().__init__()

        # hold onto the running averages for the mean and variance
        self.norm = torch.nn.ParameterDict({
            "mean": P(torch.zeros(num_features, device=self.device),
                      requires_grad=False),
            "var": P(torch.zeros(num_features, device=self.device),
                     requires_grad=False),
            "count": P(torch.zeros(1, device=self.device, dtype=torch.long),
                       requires_grad=False)})

    def forward(self, x: torch.Tensor) -> torch.Tensor: # pylint: disable=arguments-differ
        """
        Forward pass for InputNorm module

        Args:
            x: Tensor to normalize
        """

        # update running average during first epoch
        if self.training and not self.current_epoch:

            n1, m1, v1 = self.norm["count"], self.norm["mean"], self.norm["var"]
            n2 = x.shape[0]
            m2 = x.mean(dim=0)
            v2 = x.var(dim=0)

            n = n1 + n2
            mean = ((n1*m1) + (n2*m2)) / n
            d1 = v1 + (mean - m1).square()
            d2 = v2 + (mean - m2).square()
            var = ((n1*d1) + (n2*d2)) / n

            self.norm["count"] = P(n, requires_grad=False)
            self.norm["mean"] = P(mean, requires_grad=False)
            self.norm["var"] = P(var, requires_grad=False)

        # return normalized tensor
        return (x - self.norm["mean"][None, :]) / (self.norm["var"][None, :] + 1e-5).sqrt()
