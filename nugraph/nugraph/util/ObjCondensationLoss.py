import torch
from torch import Tensor

class ObjCondensationLoss(torch.nn.Module):
    def __init__(self, S_b: float = 1.0, q_min: float = 0.5):
        super().__init__()
        self.S_b = S_b
        self.q_min = q_min

    def background_loss(self, beta: Tensor, y: Tensor) -> Tensor:
        K = y.max() + 1
        n_i = (y == -1)
        M_ik = torch.zeros((y.size(0), K), device=y.device).long()
        M_ik[~n_i,:] = torch.nn.functional.one_hot(y[~n_i], num_classes=K)
        beta_ak = (beta[:,None] * M_ik).max(dim=0).values
        N_b = n_i.sum()
        L_beta_1 = (1 - beta_ak).sum() / K
        L_beta_2 = (self.S_b / N_b) * (n_i * beta).sum()
        L_beta = torch.sum(L_beta_1 + L_beta_2)
        return L_beta
    
    def potential_loss(self, x: Tensor, beta: Tensor, y: Tensor) -> Tensor:
        K = y.max() + 1
        n_i = (y == -1)
        M_ik = torch.zeros((y.size(0), K), device=y.device).long()
        M_ik[~n_i,:] = torch.nn.functional.one_hot(y[~n_i], num_classes=K)
        q_i = beta.atanh().square() + self.q_min
        q_max = (q_i[:,None] * M_ik).max(dim=0)
        q_ak = q_max.values
        idx_ak = q_max.indices
        x_a = x[idx_ak]
        x_diff = (x[:,None,:] - x_a[None,:,:]).square().sum(dim=2)
        x_inv = 1 - x_diff
        x_inv[(x_inv < 0)] = 0
        V_k_a = x_diff[None,:] * q_ak
        V_k_r = x_inv[None, :] * q_ak
        N=y.size(0)
        L_v_1=M_ik*(V_k_a)
        L_v_2=(1-M_ik)*(V_k_r)
        L_v=((L_v_1+L_v_2).sum(dim=2) * q_i).sum () / N
        return L_v

    def forward(self, x: tuple[Tensor, Tensor], y: Tensor) -> Tensor:
        x, beta = x
        return self.background_loss(beta, y) + self.potential_loss(x, beta, y)