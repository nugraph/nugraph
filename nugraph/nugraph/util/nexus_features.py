"""Nexus features transform"""
import torch
from torch_geometric.transforms import BaseTransform
from torch import cat, stack


class NexusFeatures(BaseTransform):
    '''
    Add spacepoint quality features derived from the contributing hits.

    Computes delta_T (max drift-time spread across planes) and chi2 (weighted
    time consistency) for each spacepoint, then stores [delta_T, chi2, x, y, z]
    as the spacepoint feature vector.

    Works in both data formats:
    - per-plane: data[p] node stores exist (NG2, or before NG3 Transform)
    - merged: data["hit"] with data["hit"].plane plane index (after NG3 Transform)

    Args:
        planes: Tuple of detector plane names
        positions: Whether to append the spacepoint position (x, y, z)
    '''

    def __init__(self, planes: list[str], positions: bool = True):
        super().__init__()
        self.planes = planes
        self.positions = positions

    @staticmethod
    def _scatter_to_sp(t: torch.Tensor, s: torch.Tensor,
                       sp_idx: torch.Tensor, n_sp: int):
        '''
        Scatter hit times/sigmas to one per-spacepoint value using
        sigma-weighted aggregation.

        Returns (t_sp, s_sp) each of shape [n_sp]. Spacepoints with no
        contributing hit from this plane get t=0, s=1e6 so they carry
        negligible weight in the cross-plane chi2.
        '''
        s = s.clamp(min=1e-6)
        w = 1.0 / s.square()
        sum_wt = t.new_zeros(n_sp).index_add(0, sp_idx, w * t)
        sum_w  = t.new_zeros(n_sp).index_add(0, sp_idx, w)
        has_hit = sum_w > 0
        t_sp = torch.where(has_hit, sum_wt / sum_w.clamp(min=1e-6), sum_wt)
        s_sp = torch.where(has_hit, 1.0 / sum_w.clamp(min=1e-6).sqrt(),
                           torch.full_like(sum_w, 1e6))
        return t_sp, s_sp

    def forward(self, data: 'pyg.data.HeteroData') -> 'pyg.data.HeteroData':
        n_sp = data['sp'].num_nodes
        times, sigmas = [], []

        if 'hit' in data.node_types:
            # merged format (after NG3 Transform): all planes in data["hit"]
            # with plane membership in data["hit"].plane (integer 0/1/2/...)
            edge_index = data['hit', 'nexus', 'sp'].edge_index
            hit_idx_all, sp_idx_all = edge_index[0], edge_index[1]
            plane_of_hit = data['hit'].plane[hit_idx_all]

            t_all = data['hit'].pos[hit_idx_all, 1]   # drift time always col 1 of pos

            # rms column depends on whether pos has already been prepended to x
            # by NG3 Transform's final cat((pos, x)) step:
            #   before cat: x = [integral, rms, plane_idx]         → rms at col 1
            #   after  cat: x = [wire, time, integral, rms, ...]   → rms at col n_pos+1
            n_pos = data['hit'].pos.shape[1]
            rms_col = n_pos + 1 if data['hit'].x.shape[1] > n_pos + 2 else 1
            s_all = data['hit'].x[hit_idx_all, rms_col]

            for i in range(len(self.planes)):
                mask = plane_of_hit == i
                t_sp, s_sp = self._scatter_to_sp(
                    t_all[mask], s_all[mask], sp_idx_all[mask], n_sp)
                times.append(t_sp)
                sigmas.append(s_sp)
        else:
            # per-plane format: each plane has its own node store
            for p in self.planes:
                edge_index = data[p, 'nexus', 'sp'].edge_index
                hit_idx, sp_idx = edge_index[0], edge_index[1]
                t = data[p].pos[hit_idx, 1]   # drift time for each hit
                s = data[p].x[hit_idx, 1]     # rms for each hit
                t_sp, s_sp = self._scatter_to_sp(t, s, sp_idx, n_sp)
                times.append(t_sp)
                sigmas.append(s_sp)

        times  = stack(times, dim=1)   # [n_sp, n_planes]
        sigmas = stack(sigmas, dim=1)  # [n_sp, n_planes]

        # delta_T: max drift-time spread, only over planes that have a hit
        # (missing-plane sentinels carry t=0 which would distort the range)
        has_data = sigmas < 1e5
        t_max = times.masked_fill(~has_data, float('-inf')).amax(dim=1)
        t_min = times.masked_fill(~has_data, float( 'inf')).amin(dim=1)
        delta_T = (t_max - t_min).clamp(min=0.).unsqueeze(1)

        # chi2: weighted time consistency across planes; missing planes (w≈0)
        # contribute negligibly
        w2 = 1.0 / sigmas.square()
        t_weigh_avg = ((w2 * times).sum(dim=1, keepdim=True)
                       / w2.sum(dim=1, keepdim=True).clamp(min=1e-6))
        chi2 = (w2 * (times - t_weigh_avg).square()).sum(dim=1).unsqueeze(1)

        # spacepoint features: [delta_T, chi2] (+ [x, y, z])
        feats = [delta_T, chi2] + ([data['sp'].pos] if self.positions else [])
        data['sp'].x = cat(feats, dim=1)

        return data
