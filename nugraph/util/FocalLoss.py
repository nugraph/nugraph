import torch

class FocalLoss(torch.nn.Module):
    def __init__(self,
                 weight: torch.Tensor = None,
                 gamma: float = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.celoss = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        l = self.celoss(x, y)
        if self.gamma is not None:
            pt = x.softmax(dim=1).gather(1, y.view(-1,1)).view(-1)
            l *= (1 - pt) ** self.gamma
        if self.reduction == 'none':
            return l
        elif self.reduction == 'mean':
            return l.mean()
        elif self.reduction == 'sum':
            return l.sum()
        else:
            raise Exception(f'reduction \'{self.reduction}\' not recognised!')