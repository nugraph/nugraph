"""Object condensation loss function"""
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_scatter import scatter_max

class ObjCondensationLoss(torch.nn.Module):
    def __init__(self, s_b: float = 1.0, q_min: float = 0.5):
        super().__init__()
        self.s_b = s_b
        self.q_min = q_min
    
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

    def forward(self, data: HeteroData, y: Tensor) -> Tensor:

        device = data["hit"].x.device

        # hit information
        x = data["hit"].ox
        f = data["hit"].of

        # true instances
        n_true = data["particle-truth"].num_nodes
        e_true = data["hit", "cluster-truth", "particle-truth"].edge_index

        bkg_mask = y == -1
        n_bkg = bkg_mask.sum()

        # check inputs
        if not n_true:
            raise RuntimeError(("Cannot compute object condensation loss "
                                "when there are no true instances!"))
        if not n_bkg:
            raise RuntimeError(("Cannot compute object condensation loss "
                                "when there is no true background!"))

        # determine which hit is the condensation point
        # for each true instance, and get their beta values
        # f_true and their hit indices i_true
        i, j = e_true
        f_centers = torch.zeros(n_true, device=device)
        f_centers, i_centers = scatter_max(f[i], j, out=f_centers)
        i_centers = i[i_centers]

        # calculate the two background loss terms, b1 and b2
        b1 = 1 - (f_centers.sum() / n_true)
        b2 = (self.s_b / n_bkg) * f[bkg_mask].sum()

        return b1 + b2 + self.potential_loss(x, f, y)
