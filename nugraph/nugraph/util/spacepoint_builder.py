"""Spacepoint candidates built from hit crossings"""
import torch
from torch_geometric.transforms import BaseTransform

from pynuml.data import NuGraphData


class SpacePointBuilder(BaseTransform):
    """
    Replace an event's spacepoints with candidates built from its hits

    Each hit measures position along two known axes of its plane (see
    --plane-geometry): the wire coordinate and the drift coordinate, each with
    a 3D direction, its value at the detector origin and a resolution. Two hits
    on different planes whose drift coordinates agree within window cross at
    the 3D point that best fits all four measurements (weighted least
    squares); crossings outside the active volume are dropped. A crossing of
    the first two planes with a third-plane hit whose wire coordinate agrees
    within tolerance, and whose drift coordinate is in window, is a triplet,
    refitted from all three hits. Candidates are the triplets, plus the
    two-plane crossings of hits that no triplet uses, which is the provided
    MicroBooNE product's mix. With best, a candidate is kept only when it is
    among the best best candidates of at least one of its hits, scoring
    triplets by wire residual / tolerance + drift spread / window and doublets
    by 1 + drift difference / window (lower is better): a hit that spans many
    ticks, or a track parallel to the wire planes, otherwise gives many
    overlapping candidates of one particle. If there are more than cap, the
    triplets are kept first, then the doublets with the closest drift
    coordinates.

    The candidates replace data["sp"], the hit-to-spacepoint links and, when
    present, the spacepoint-to-event links, so the
    spacepoint graph, truth labels and features are built on them as on the
    provided spacepoints. Expects the merged hit format (after the NuGraph3
    transform), with hit positions (wire coordinate, drift coordinate) in cm.

    Args:
        geometry: Ten numbers per plane (wire axis direction, origin and
            resolution, then the drift axis's), as from --plane-geometry
        volume: Active volume as ((x, y, z) minimum, (x, y, z) maximum), cm
        window: Largest drift-coordinate difference between a candidate's hits, cm
        tolerance: Largest third-plane wire-coordinate residual of a triplet, cm
        cap: Largest number of candidates per event
        best: Candidates kept per hit (0 keeps all)
    """
    def __init__(self, geometry: list[list[float]], volume: list[list[float]],
                 window: float = 0.3, tolerance: float = 0.45, cap: int = 20000,
                 best: int = 0):
        super().__init__()
        self.geometry = torch.tensor(geometry, dtype=torch.float64)
        self.volume = torch.tensor(volume, dtype=torch.float64)
        self.window = window
        self.tolerance = tolerance
        self.cap = cap
        self.best = best

    def measurements(self, data: NuGraphData) -> tuple:
        """Each hit's two measurement directions, values from the origin and weights"""
        h = data["hit"]
        g = self.geometry[h.plane]
        pos = h.pos.double()
        dirs = torch.stack((g[:, 0:3], g[:, 5:8]), dim=1)                   # (hits, 2, 3)
        vals = torch.stack((pos[:, 0] - g[:, 3], pos[:, 1] - g[:, 8]), dim=1)
        wts = torch.stack((g[:, 4], g[:, 9]), dim=1).pow(-2)
        return dirs, vals, wts

    @staticmethod
    def fit(dirs: torch.Tensor, vals: torch.Tensor, wts: torch.Tensor,
            hits: torch.Tensor) -> torch.Tensor:
        """Weighted least-squares 3D point of each row of hits (candidates, hits per candidate)"""
        d, v, w = dirs[hits].flatten(1, 2), vals[hits].flatten(1), wts[hits].flatten(1)
        a = torch.einsum("cm,cmi,cmj->cij", w, d, d)
        b = torch.einsum("cm,cmi,cm->ci", w, d, v)
        return torch.linalg.solve(a, b)

    def pairs(self, drift: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> tuple:
        """Pairs of hits from index sets i and j whose drift coordinates agree within window"""
        o = drift[j].argsort()
        sj = drift[j][o]
        lo = torch.searchsorted(sj, drift[i] - self.window)
        hi = torch.searchsorted(sj, drift[i] + self.window)
        a = torch.repeat_interleave(torch.arange(len(i)), hi - lo)
        start = torch.repeat_interleave(lo - torch.cumsum(hi - lo, 0) + (hi - lo), hi - lo)
        b = o[torch.arange(len(a)) + start] if len(a) else a
        return i[a], j[b]

    def keep_best(self, cands: list[torch.Tensor], scores: list[torch.Tensor]) -> list[torch.Tensor]:
        """Masks of the candidates (rows of hit indices) among the best of at least one of their hits"""
        n = [len(c) for c in cands]
        cand = torch.cat([torch.arange(len(c)).repeat_interleave(c.size(1)) + sum(n[:i])
                          for i, c in enumerate(cands)])
        hit = torch.cat([c.flatten() for c in cands])
        score = torch.cat(scores)[cand]
        o = score.argsort(stable=True)
        o = o[hit[o].argsort(stable=True)]
        hs = hit[o]
        rank = torch.arange(len(hs)) - torch.searchsorted(hs, hs)
        keep = torch.zeros(sum(n), dtype=torch.bool)
        keep[cand[o][rank < self.best]] = True
        return list(keep.split(n))

    def inside(self, pos: torch.Tensor) -> torch.Tensor:
        return ((pos >= self.volume[0]) & (pos <= self.volume[1])).all(dim=1)

    def forward(self, data: NuGraphData) -> NuGraphData:
        dirs, vals, wts = self.measurements(data)
        plane = data["hit"].plane
        drift = vals[:, 1]
        idx = [torch.nonzero(plane == p).squeeze(1) for p in range(len(self.geometry))]

        # two-plane crossings of every pair of planes
        doublets = []
        for p in range(len(idx)):
            for q in range(p + 1, len(idx)):
                hp, hq = self.pairs(drift, idx[p], idx[q])
                hits = torch.stack((hp, hq), dim=1)
                ok = self.inside(self.fit(dirs, vals, wts, hits)) if len(hits) else hp.new_zeros(0, dtype=torch.bool)
                doublets.append(hits[ok])

        # triplets: crossings of the first two planes with a third-plane hit
        triplets = torch.zeros(0, 3, dtype=torch.long)
        if len(idx) >= 3 and len(doublets[0]):
            pq = doublets[0]
            r = self.fit(dirs, vals, wts, pq)
            third = idx[2]
            wire = vals[third, 0]
            o = wire.argsort()
            pred = r @ self.geometry[2, 0:3].double()
            lo = torch.searchsorted(wire[o], pred - self.tolerance)
            hi = torch.searchsorted(wire[o], pred + self.tolerance)
            k = torch.repeat_interleave(torch.arange(len(pq)), hi - lo)
            start = torch.repeat_interleave(lo - torch.cumsum(hi - lo, 0) + (hi - lo), hi - lo)
            h3 = third[o[torch.arange(len(k)) + start]] if len(k) else k
            t = (drift[pq[k, 0]] + drift[pq[k, 1]]) / 2
            ok = (drift[h3] - t).abs() < self.window
            triplets = torch.cat((pq[k[ok]], h3[ok, None]), dim=1)
            residual = (pred[k] - vals[h3, 0]).abs()[ok]

        # doublets of hits no triplet uses, nearest drift agreement first
        used = torch.zeros(len(plane), dtype=torch.bool)
        used[triplets.flatten()] = True
        doublets = torch.cat(doublets) if doublets else torch.zeros(0, 2, dtype=torch.long)
        doublets = doublets[~used[doublets].any(dim=1)]
        if self.best and len(triplets) + len(doublets):
            d = drift[triplets]
            t_score = residual / self.tolerance + (d.max(dim=1).values - d.min(dim=1).values) / self.window \
                if len(triplets) else drift.new_zeros(0)
            d_score = 1 + (drift[doublets[:, 0]] - drift[doublets[:, 1]]).abs() / self.window
            kt, kd = self.keep_best([triplets, doublets], [t_score, d_score])
            triplets, doublets = triplets[kt], doublets[kd]
        doublets = doublets[(drift[doublets[:, 0]] - drift[doublets[:, 1]]).abs().argsort()]
        n_dbl = max(0, min(len(doublets), self.cap - len(triplets)))
        triplets, doublets = triplets[:self.cap], doublets[:n_dbl]

        pos = torch.cat((self.fit(dirs, vals, wts, triplets) if len(triplets) else torch.zeros(0, 3, dtype=torch.float64),
                         self.fit(dirs, vals, wts, doublets) if len(doublets) else torch.zeros(0, 3, dtype=torch.float64)))
        hit = torch.cat((triplets.flatten(), doublets.flatten()))
        sp = torch.cat((torch.arange(len(triplets)).repeat_interleave(3),
                        torch.arange(len(doublets)).repeat_interleave(2) + len(triplets)))

        for key in list(data["sp"].keys()):
            del data["sp"][key]
        data["sp"].pos = pos.float()
        data["sp"].num_nodes = len(pos)
        data["hit", "nexus", "sp"].edge_index = torch.stack((hit, sp), dim=0)
        # the model transform links the file's spacepoints to the event node
        if ("sp", "in", "evt") in data.edge_types:
            data["sp", "in", "evt"].edge_index = torch.stack(
                (torch.arange(len(pos)), torch.zeros(len(pos), dtype=torch.long)), dim=0)
        return data
