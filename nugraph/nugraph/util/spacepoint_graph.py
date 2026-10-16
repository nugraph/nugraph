"""Spacepoint graph transform"""
import math

import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import BaseTransform

from pynuml.data import NuGraphData

GRAPHS = ("knn", "gabriel", "gabriel+stubs")
IONI = (2, 3, 4, 5)   # pynuml start-process codes of muIoni, hIoni, eIoni, ionIoni

class SpacePointGraph(BaseTransform):
    """
    Add a 3D graph over spacepoint nodes

    With the "knn" graph, each spacepoint receives edges from its k nearest
    neighbours by Euclidean distance. The "gabriel" graph joins two
    spacepoints when no other spacepoint lies inside the sphere that has them
    as a diameter, searched among each spacepoint's candidates nearest
    neighbours; it is a subgraph of the Delaunay triangulation that keeps
    chains of points along a trajectory, including across gaps.
    "gabriel+stubs" adds edges along local track stubs: neighbours in a shell
    around each spacepoint vote for their direction on a grid of directions
    (a local radial Hough transform), peaks in the votes are stubs, and each
    spacepoint is joined to its nearest neighbours inside a narrow cone along
    each stub. Gabriel and stub edges are undirected, stored in both
    directions. Graphs are built within each event, so the transform also
    applies to a batch of events, on any device.

    If the graph carries true hit instance labels, each spacepoint is
    labelled with the instance shared by all of its hits (-1 if its hits come
    from more than one instance or from background), each edge is labelled
    1 if both spacepoints belong to the same instance, and each labelled
    spacepoint gets the local direction of its instance (y_direction), the
    principal axis of itself and its same-instance neighbours among its k
    nearest neighbours; spacepoints with fewer than two such neighbours get a
    zero vector. Each spacepoint also gets the shape of its k-nearest-neighbour
    neighbourhood (local_shape): the six independent entries of the
    covariance of its neighbours' offsets, divided by its trace, independent
    of the graph used for message passing. With particle ancestry, spacepoints of delta rays (electrons
    made by ionisation) shorter than merge_delta can be labelled with the
    delta ray's parent instead.

    Args:
        k: Number of nearest neighbours to connect each spacepoint to, and to
            find each spacepoint's local direction from
        chunk_size: Number of spacepoints to compute distances for at once,
            bounding peak memory to chunk_size * n_sp
        majority: Also label spacepoints with hits in three planes, two from
            the same instance and an odd hit that carries truth (another
            instance, or a diffuse deposit), with the majority instance
        graph: Graph type ("knn", "gabriel" or "gabriel+stubs")
        candidates: Number of nearest neighbours searched for Gabriel edges
        stub_radius: Outer radius of the shell of neighbours voting for stub
            directions, and reach of stub edges (at least 15 cm)
        stub_candidates: Number of nearest neighbours searched for stubs
        extra_knn: Also join each spacepoint to this many nearest neighbours
            in the Gabriel graphs, so that a track does not rest on a single
            chain of edges
        merge_delta: Label delta rays shorter than this (cm, from their true
            start and end) as their parent particle; 0 to keep them apart
    """
    def __init__(self, k: int, chunk_size: int = 1024, majority: bool = True,
                 graph: str = "knn", candidates: int = 64, stub_radius: float = 24.,
                 stub_candidates: int = 256, merge_delta: float = 0., extra_knn: int = 0):
        super().__init__()
        if graph not in GRAPHS:
            raise ValueError(f'"{graph}" is not a valid spacepoint graph option.')
        self.k = k
        self.chunk_size = chunk_size
        self.majority = majority
        self.graph = graph
        self.candidates = candidates
        self.stub_radius = stub_radius
        self.stub_candidates = stub_candidates
        self.merge_delta = merge_delta
        self.extra_knn = extra_knn

        # local Hough transform: directions on a Fibonacci lattice over the
        # sphere, and the nearest grid directions of each
        m = torch.arange(400, dtype=torch.float) + 0.5
        z = 1 - 2 * m / 400
        phi = math.pi * (1 + 5 ** 0.5) * m
        r = (1 - z ** 2).sqrt()
        self.grid = torch.stack((r * phi.cos(), r * phi.sin(), z), dim=1)
        self.grid_nbr = (self.grid @ self.grid.T).argsort(dim=1, descending=True)[:, 1:9]

    def forward(self, data: NuGraphData) -> NuGraphData:
        """
        Apply transform to add spacepoint edges and truth labels

        Args:
            data: NuGraph data object (or batch of objects) to transform
        """

        sp = data["sp"]
        n = sp.num_nodes
        dev = sp.pos.device
        batch = sp.batch if "batch" in sp else torch.zeros(n, dtype=torch.long, device=dev)
        arange = torch.arange(n, device=dev)

        # nearest neighbours within each event; missing neighbours are -1
        k = {"knn": self.k, "gabriel": max(self.k, self.candidates),
             "gabriel+stubs": max(self.k, self.candidates, self.stub_candidates)}[self.graph]
        nbr, dist = self.neighbours(sp.pos, batch, k)
        knn = nbr[:, :self.k]
        ok = knn >= 0
        knn_index = torch.stack((knn[ok], arange[:, None].expand_as(knn)[ok]), dim=0)
        d = sp.pos[knn_index[0]] - sp.pos[knn_index[1]]
        cov = torch.zeros(n, 3, 3, dtype=sp.pos.dtype, device=dev).index_add(
            0, knn_index[1], d[:, :, None] * d[:, None, :])
        cov = cov / cov.diagonal(dim1=1, dim2=2).sum(dim=1).clamp(min=1e-6)[:, None, None]
        r, c = torch.triu_indices(3, 3, device=dev)
        sp.local_shape = cov[:, r, c]
        if self.graph == "knn":
            edge_index = knn_index
        else:
            pairs = [self.gabriel(sp.pos, nbr[:, :self.candidates])]
            if self.extra_knn:
                extra = nbr[:, :self.extra_knn]
                ok = extra >= 0
                pairs.append(torch.stack((arange[:, None].expand_as(extra)[ok], extra[ok]), dim=0))
            if self.graph == "gabriel+stubs":
                stub_pairs, _, stub_votes = self.stubs(sp.pos, nbr[:, :self.stub_candidates],
                                                       dist[:, :self.stub_candidates],
                                                       r_max=self.stub_radius,
                                                       reach=max(15., self.stub_radius),
                                                       return_stubs=True)
                pairs.append(stub_pairs)
                # number of stubs at each spacepoint: one along a single track,
                # more where tracks meet, none in diffuse activity
                sp.stub_count = (stub_votes > 0).sum(dim=1).float()
            pairs = torch.cat(pairs, dim=1)
            pairs = torch.cat((pairs, pairs.flip(0)), dim=1)
            edge_index = torch.unique(pairs[0] * n + pairs[1])
            edge_index = torch.stack((edge_index // n, edge_index % n), dim=0)
        edge = data["sp", "sp3d", "sp"]
        edge.edge_index = edge_index

        # truth labels, if true hit instances are available
        if ("hit", "cluster-truth", "particle-truth") in data.edge_types:
            y_i = data.y_i()
            hit, s = data["hit", "nexus", "sp"].edge_index
            lo = torch.full((n,), torch.iinfo(torch.long).max, dtype=torch.long, device=dev)
            hi = torch.full((n,), -1, dtype=torch.long, device=dev)
            lo = lo.scatter_reduce(0, s, y_i[hit], reduce="amin")
            hi = hi.scatter_reduce(0, s, y_i[hit], reduce="amax")
            sp.y_instance = torch.where((lo == hi) & (lo >= 0), lo, -1)
            if self.majority:
                sp.y_instance = self.majority_labels(sp.y_instance, y_i[hit],
                                                     data["hit"].y_semantic[hit], s)
            if self.merge_delta and "parent_g4_id" in data["particle-truth"]:
                sp.y_instance = self.merge_short_delta_rays(data, sp.y_instance)
            # majority semantic class of each spacepoint's labelled hits (-1 if none)
            sem_hit = data["hit"].y_semantic[hit]
            labelled = sem_hit >= 0
            n_cls = int(sem_hit.max()) + 1 if labelled.any() else 1
            votes = torch.zeros(n, n_cls, device=dev).index_put_(
                (s[labelled], sem_hit[labelled]), torch.ones(int(labelled.sum()), device=dev), accumulate=True)
            sp.y_semantic = torch.where(votes.sum(dim=1) > 0, votes.argmax(dim=1), -1)
            y = sp.y_instance
            edge.y = ((y[edge_index[0]] >= 0) & (y[edge_index[0]] == y[edge_index[1]])).long()

            # local instance direction: principal axis of each spacepoint and
            # its same-instance nearest neighbours
            src, dst = knn_index[:, (y[knn_index[0]] >= 0) & (y[knn_index[0]] == y[knn_index[1]])]
            pos = torch.cat((sp.pos[src], sp.pos), dim=0).double()
            idx = torch.cat((dst, arange), dim=0)
            count = torch.zeros(n, dtype=pos.dtype, device=dev).index_add(
                0, idx, torch.ones_like(idx, dtype=pos.dtype))
            mean = torch.zeros(n, 3, dtype=pos.dtype, device=dev).index_add(0, idx, pos) / count[:, None]
            d = pos - mean[idx]
            cov = torch.zeros(n, 3, 3, dtype=pos.dtype, device=dev).index_add(
                0, idx, d[:, :, None] * d[:, None, :])
            direction = torch.linalg.eigh(cov).eigenvectors[:, :, -1]
            ok = (y >= 0) & (count >= 3)
            sp.y_direction = torch.where(ok[:, None], direction, 0.).float()

        # in a batch, instance labels are numbered within each event, and the
        # new attributes are registered so that the batch can be split
        if isinstance(data, Batch):
            # pylint: disable=protected-access
            zeros = torch.zeros(data.num_graphs, device=dev)
            counts = torch.zeros(data.num_graphs, dtype=torch.long, device=dev).index_add(
                0, batch[edge_index[1]], torch.ones_like(edge_index[1]))
            data._slice_dict[edge._key] = {"edge_index": torch.cat((counts.new_zeros(1), counts.cumsum(0)))}
            data._inc_dict[edge._key] = {"edge_index": sp.ptr[:-1, None, None].repeat(1, 2, 1)}
            data._slice_dict["sp"]["local_shape"] = sp.ptr
            data._inc_dict["sp"]["local_shape"] = zeros
            if "stub_count" in sp:
                data._slice_dict["sp"]["stub_count"] = sp.ptr
                data._inc_dict["sp"]["stub_count"] = zeros
            if "y" in edge:
                data._slice_dict[edge._key]["y"] = data._slice_dict[edge._key]["edge_index"]
                data._inc_dict[edge._key]["y"] = zeros
            if "y_instance" in sp:
                offset = data["particle-truth"].ptr[:-1][batch]
                sp.y_instance = torch.where(sp.y_instance >= 0, sp.y_instance - offset, -1)
                for attr in ("y_instance", "y_direction", "y_semantic"):
                    data._slice_dict["sp"][attr] = sp.ptr
                    data._inc_dict["sp"][attr] = zeros

        return data

    def neighbours(self, pos: torch.Tensor, batch: torch.Tensor,
                   k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Indices of and distances to each spacepoint's k nearest neighbours in
        its own event, nearest first; missing neighbours get index -1 and
        infinite distance

        Args:
            pos: Spacepoint positions
            batch: Event index of each spacepoint
            k: Number of neighbours
        """
        n = pos.size(0)
        k = min(k, n - 1)
        if k <= 0:
            return (torch.empty((n, 0), dtype=torch.long, device=pos.device),
                    torch.empty((n, 0), device=pos.device))
        # spacepoints of each event are contiguous, so each chunk is compared
        # only against the spacepoints of the events it spans
        ptr = torch.searchsorted(batch, torch.arange(int(batch[-1]) + 2, device=pos.device)).tolist()
        nbrs, dists = [], []
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            lo, hi = ptr[int(batch[start])], ptr[int(batch[end - 1]) + 1]
            dist = torch.cdist(pos[start:end], pos[lo:hi])
            dist[batch[start:end, None] != batch[None, lo:hi]] = torch.inf
            dist[torch.arange(end - start), torch.arange(start - lo, end - lo)] = torch.inf
            dist, nbr = dist.topk(min(k, hi - lo), largest=False)
            nbr = torch.where(dist.isfinite(), nbr + lo, -1)
            if nbr.size(1) < k:
                nbr = torch.nn.functional.pad(nbr, (0, k - nbr.size(1)), value=-1)
                dist = torch.nn.functional.pad(dist, (0, k - dist.size(1)), value=torch.inf)
            nbrs.append(nbr)
            dists.append(dist)
        return torch.cat(nbrs, dim=0), torch.cat(dists, dim=0)

    def gabriel(self, pos: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
        """
        Gabriel edges among nearest-neighbour candidates: spacepoint i and its
        neighbour j are joined unless a nearer neighbour s of i lies inside
        the sphere with diameter ij, that is, sees i and j at an obtuse angle

        Args:
            pos: Spacepoint positions
            nbr: Nearest-neighbour indices, nearest first, -1 if missing
        """
        pairs = []
        for start in range(0, nbr.size(0), self.chunk_size):
            c = nbr[start:start + self.chunk_size]
            ok = c >= 0
            rel = pos[c.clamp(min=0)] - pos[start:start + c.size(0), None]
            dot = rel @ rel.transpose(1, 2)              # dot[i, t, s] = rel_t . rel_s
            norm2 = dot.diagonal(dim1=1, dim2=2)          # |rel_s|^2
            inside = (norm2[:, None, :] - dot) < -1e-5 * norm2[:, :, None]
            inside &= ok[:, None, :] & torch.ones_like(dot, dtype=torch.bool).tril(-1)
            keep = ok & ~inside.any(dim=2)
            i = torch.arange(start, start + c.size(0), device=pos.device)[:, None].expand_as(c)
            pairs.append(torch.stack((i[keep], c[keep]), dim=0))
        return torch.cat(pairs, dim=1)

    def stubs(self, pos: torch.Tensor, nbr: torch.Tensor, dist: torch.Tensor,
              r_min: float = 1., r_max: float = 8., kappa: float = 30.,
              min_votes: float = 3., sep_deg: float = 25., max_stubs: int = 6,
              cone_deg: float = 10., reach: float = 15., per_stub: int = 2,
              return_stubs: bool = False) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Edges along local track stubs. Neighbours between r_min and r_max of
        each spacepoint vote for their direction on the direction grid with a
        von Mises-Fisher kernel of concentration kappa; grid directions with
        at least min_votes that are local maxima are stubs, taken in order of
        votes and at least sep_deg apart. Each spacepoint is joined to its
        per_stub nearest neighbours within reach inside a cone of half-angle
        cone_deg along each stub.

        Args:
            pos: Spacepoint positions
            nbr: Nearest-neighbour indices, nearest first, -1 if missing
            dist: Nearest-neighbour distances
            return_stubs: Also return each spacepoint's stub directions and
                votes, in order of votes (zero for slots without a stub)
        """
        grid = self.grid.to(pos.device)
        grid_nbr = self.grid_nbr.to(pos.device)
        cos_sep = math.cos(math.radians(sep_deg))
        cos_cone = math.cos(math.radians(cone_deg))
        pairs, stub_dirs, stub_votes = [], [], []
        for start in range(0, nbr.size(0), self.chunk_size):
            c, d = nbr[start:start + self.chunk_size], dist[start:start + self.chunk_size]
            ok = c >= 0
            rel = pos[c.clamp(min=0)] - pos[start:start + c.size(0), None]
            u = rel / d.clamp(min=1e-6)[:, :, None]
            shell = ok & (d >= r_min) & (d < r_max)
            # votes are the costliest step: in half precision on GPU, with the
            # kernel evaluated in place and summed over the shell by a matmul
            half = torch.float16 if pos.is_cuda else torch.float32
            votes = (u.to(half) @ grid.to(half).T).sub_(1).mul_(kappa).exp_()
            score = (shell.to(half)[:, None, :] @ votes).squeeze(1).float()
            peak = (score >= min_votes) & (score >= score[:, grid_nbr].amax(dim=2))
            top, cand = torch.where(peak, score, -1.).topk(min(max_stubs, grid.size(0)), dim=1)
            dirs = grid[cand]
            chosen = torch.zeros_like(top, dtype=torch.bool)
            for p in range(top.size(1)):
                close = ((dirs @ dirs[:, p, :, None]).squeeze(2) >= cos_sep) & chosen
                chosen[:, p] = (top[:, p] > 0) & ~close.any(dim=1)
            inside = ((u @ dirs.transpose(1, 2)) >= cos_cone).transpose(1, 2)
            inside &= (ok & (d > 0) & (d < reach))[:, None, :] & chosen[:, :, None]
            take = inside & (inside.cumsum(dim=2) <= per_stub)
            i = torch.arange(start, start + c.size(0), device=pos.device)[:, None, None].expand_as(take)
            pairs.append(torch.stack((i[take], c[:, None, :].expand_as(take)[take]), dim=0))
            if return_stubs:
                stub_dirs.append(dirs * chosen[:, :, None])
                stub_votes.append(top * chosen)
        if return_stubs:
            return torch.cat(pairs, dim=1), torch.cat(stub_dirs), torch.cat(stub_votes)
        return torch.cat(pairs, dim=1)

    def merge_short_delta_rays(self, data: NuGraphData, y: torch.Tensor) -> torch.Tensor:
        """
        Relabel spacepoints of delta rays shorter than merge_delta with their
        parent, following chains of delta rays up to the first longer
        ancestor, within each event

        Args:
            data: NuGraph data object (or batch of objects) with particle ancestry
            y: Spacepoint instance labels (particle-truth node indices)
        """
        pt = data["particle-truth"]
        n = pt.num_nodes
        dev = y.device
        event = pt.batch if "batch" in pt else torch.zeros(n, dtype=torch.long, device=dev)
        # find each particle's parent in its own event by (event, Geant4 ID)
        key = event * 2 ** 40 + pt.g4_id
        order = key.argsort()
        parent_key = event * 2 ** 40 + pt.parent_g4_id
        at = torch.searchsorted(key[order], parent_key).clamp(max=n - 1)
        found = key[order][at] == parent_key
        parent = torch.where(found, order[at], -1)
        length = (pt.end_position - pt.start_position).norm(dim=1)
        delta = (pt.pdg_code.abs() == 11) & torch.isin(pt.start_process, torch.tensor(IONI, device=dev))
        target = torch.arange(n, device=dev)
        short = delta & (length < self.merge_delta) & (parent >= 0)
        target[short] = parent[short]
        for _ in range(8):
            target = target[target]
        return torch.where(y >= 0, target[y.clamp(min=0)], y)

    @staticmethod
    def majority_labels(y: torch.Tensor, y_hit: torch.Tensor, sem_hit: torch.Tensor,
                        s: torch.Tensor) -> torch.Tensor:
        """
        Give unlabelled spacepoints with three hits, two from one instance and
        an odd hit that carries truth, the majority instance

        Args:
            y: Spacepoint instance labels
            y_hit: Instance of each hit-spacepoint edge's hit
            sem_hit: Semantic label of each hit-spacepoint edge's hit
            s: Spacepoint of each hit-spacepoint edge
        """
        n = y.size(0)
        dev = y.device
        order = torch.argsort(s, stable=True)
        s, y_hit, sem_hit = s[order], y_hit[order], sem_hit[order]
        first = torch.searchsorted(s, s, side="left")
        slot = torch.arange(s.size(0), device=dev) - first
        inst = torch.full((n, 3), -2, dtype=torch.long, device=dev)
        sem = torch.full((n, 3), -2, dtype=torch.long, device=dev)
        ok = slot < 3
        inst[s[ok], slot[ok]] = y_hit[ok]
        sem[s[ok], slot[ok]] = sem_hit[ok]
        nhit = torch.zeros(n, dtype=torch.long, device=dev).index_add(0, s, torch.ones_like(s))
        a, b, c = inst.unbind(dim=1)
        major = torch.where(a == b, a, torch.where(a == c, a, torch.where(b == c, b, -1)))
        odd = torch.where(a == b, 2, torch.where(a == c, 1, 0))
        odd_inst = inst.gather(1, odd[:, None]).squeeze(1)
        odd_sem = sem.gather(1, odd[:, None]).squeeze(1)
        relabel = (y < 0) & (nhit == 3) & (major >= 0) & (odd_inst != major) & (odd_sem >= 0)
        return torch.where(relabel, major, y)
