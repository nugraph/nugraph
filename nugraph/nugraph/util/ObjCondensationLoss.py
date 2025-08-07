"""Object condensation loss function"""
import torch
from torch_scatter import scatter_max

T = torch.Tensor

class ObjCondensationLoss(torch.nn.Module):
    def __init__(self, s_b: float = 1.0, q_min: float = 0.5):
        super().__init__()
        self.s_b = s_b
        self.q_min = q_min

    def forward(self, x: T, f: T, y_i: T, y_s: T, n_true: int, e_true: T) -> T:
        device = x.device
        dtype = x.dtype

        # hit information
        n_hit = x.size(0)

        # check inputs
        if not n_true:
            raise RuntimeError(("Cannot compute object condensation loss "
                                "when there are no true instances!"))

        # determine which hit is the condensation point for each true instance,
        # and get beta values (f_centers) and hit indices (centers)
        e_h, e_p = e_true
        f_centers = torch.zeros(n_true, dtype=dtype, device=device)
        f_centers, centers = scatter_max(f[e_h], e_p, dim_size=n_true) #Updated dim_size argument
        centers = e_h[centers]
        #--------------------------
        print("f_centers:", f_centers.shape, f_centers.device)
        print("f_centers.sum():", f_centers.sum())
        print("n_true:", n_true)

        # Catch divide-by-zero
        if n_true == 0:
            raise RuntimeError("n_true is 0 — division invalid!")

        # Catch device mismatch
        if not isinstance(n_true, torch.Tensor):
            n_true = torch.tensor(n_true, device=f_centers.device)
        elif f_centers.device != n_true.device:
            print("Warning: device mismatch between f_centers and n_true")
            n_true = n_true.to(f_centers.device)

        # Catch NaNs
        if torch.isnan(f_centers).any():
            raise RuntimeError("f_centers contains NaNs")
        #--------------------------
        # calculate background loss terms
        b = 1 - (f_centers.sum() / n_true)
        print("b:", b)
        bkg_mask = (y_i == -1) & (y_s >= 0)
        n_bkg = bkg_mask.sum()
        if n_bkg:
            b += (self.s_b / n_bkg) * f[bkg_mask].sum()

        # calculate the charge on each hit
        q = f.atanh().square() + self.q_min

        # calculate attractive and repulsive potentials
        m_ik = torch.zeros(n_hit, n_true, dtype=torch.bool, device=device)
        m_ik[e_h, e_p] = True
        dist = (x[:, None, :] - x[centers][None, :, :]).square().sum(dim=2)
        v = torch.where(m_ik, dist, (1 - dist).clamp(0))
        v = ((v * q[centers]).sum(dim=1) * q).sum() / n_hit

        return torch.stack([b, v])