"""NuGraph2 class linear module"""
import torch

T = torch.Tensor

class ClassLinear(torch.nn.Module):
    """
    NuGraph2 module for linear transformations grouped by class
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        num_classes: Number of semantic classes
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_classes: int):
        super().__init__()

        self.num_classes = num_classes

        self.net = nn.ModuleList()
        for _ in range(num_classes):
            self.net.append(nn.Linear(in_features, out_features))

    def forward(self, x: T) -> T:
        """
        NuGraph2 class linear module forward pass

        Args:
            x: Feature tensor to transform
        """
        xs = torch.tensor_split(x, self.num_classes, dim=1)
        return torch.cat([ net(xs[i]) for i, net in enumerate(self.net) ], dim=1)
