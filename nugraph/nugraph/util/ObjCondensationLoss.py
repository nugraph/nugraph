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

    def forward(self, data: HeteroData, y: Tensor) -> Tensor:

        device = data["hit"].x.device

        # hit information
        n_hit = data["hit"].num_nodes
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

        # determine which hit is the condensation point for each true instance,
        # and get beta values (f_centers) and hit indices (centers)
        e_h, e_p = e_true
        f_centers = torch.zeros(n_true, device=device)
        f_centers, centers = scatter_max(f[e_h], e_p, out=f_centers)
        centers = e_h[centers]

        # calculate background loss terms
        b1 = 1 - (f_centers.sum() / n_true)
        b2 = (self.s_b / n_bkg) * f[bkg_mask].sum()

        # calculate the charge on each hit
        q = f.atanh().square() + self.q_min

        # calculate attractive and repulsive potentials
        m_ik = torch.zeros(n_hit, n_true, dtype=torch.bool, device=device)
        m_ik[e_h, e_p] = True
        dist = (x[:, None, :] - x[centers][None, :, :]).square().sum(dim=2)
        v = torch.where(m_ik, dist, (1 - dist).clamp(0))
        v = ((v * q[centers]).sum(dim=1) * q).sum() / n_hit

        return torch.stack([b1 + b2, v])
