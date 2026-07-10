"""Object condensation loss function"""
import torch
from torch_scatter import scatter_max

T = torch.Tensor

class ObjCondensationLoss(torch.nn.Module):
    def __init__(self, s_b: float = 1.0, q_min: float = 0.5):
        super().__init__()
        self.s_b = s_b
        self.q_min = q_min

    def l_b(self, f: T, f_centers: T, bkg_mask: T, n_true: int) -> T:
        """Calculate background loss term"""
        b = 1 - (f_centers.sum() / n_true)
        n_bkg = bkg_mask.sum()
        if n_bkg:
            b += (self.s_b / n_bkg) * f[bkg_mask].sum()
        return b

    def l_v(self, x: T, f: T, centers: T, e_h: T, e_p: T, n_true: int) -> T:
        """Calculate potential loss term"""
        device = x.device
        n_hit = x.size(0)
        q = f.atanh().square() + self.q_min
        m_ik = torch.zeros(n_hit, n_true, dtype=torch.bool, device=device)
        m_ik[e_h, e_p] = True
        dist = (x[:, None, :] - x[centers][None, :, :]).square().sum(dim=2)
        v = torch.where(m_ik, dist, (1 - dist).clamp(0))
        v = ((v * q[centers]).sum(dim=1) * q).sum() / n_hit
        return v

    def l_p(self, f: T, bkg_mask: T, l_p: T) -> T:
        """Calculate particle loss term"""
        if l_p is None:
            dtype = f.dtype
            device = f.device
            p = torch.tensor(0., dtype=dtype, device=device)
        else:
            xi = f[~bkg_mask].atanh().square()
            p = (l_p[~bkg_mask] * xi[:, None]).sum() / (xi.sum())
        return p

    def forward(self, x: T, f: T, y_i: T, y_s: T, n_true: int, e_true: T,
                l_p: T) -> T:

        device = x.device
        dtype = x.dtype

        # hit information
        n_hit = x.size(0)

        # check inputs
        if not n_true:
            return torch.zeros(3, dtype=dtype, device=device)

        # determine which hit is the condensation point for each true instance,
        # and get beta values (f_centers) and hit indices (centers)
        e_h, e_p = e_true
        f_centers = torch.zeros(n_true, dtype=dtype, device=device)
        f_centers, centers = scatter_max(f[e_h], e_p, out=f_centers)
        centers = e_h[centers]

        bkg_mask = (y_i == -1) & (y_s >= 0)

        # calculate loss terms
        b = self.l_b(f, f_centers, bkg_mask, n_true)
        v = self.l_v(x, f, centers, e_h, e_p, n_true)
        p = self.l_p(f, bkg_mask, l_p)

        return torch.stack([b, v, p])
