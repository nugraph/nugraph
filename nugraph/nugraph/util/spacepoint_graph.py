"""Spacepoint graph transform"""
import torch
from torch_geometric.transforms import BaseTransform

from pynuml.data import NuGraphData

class SpacePointGraph(BaseTransform):
    """
    Add a 3D k-nearest-neighbour graph over spacepoint nodes

    Each spacepoint receives edges from its k nearest neighbours by Euclidean
    distance. If the graph carries true hit instance labels, each spacepoint is
    labelled with the instance shared by all of its hits (-1 if its hits come
    from more than one instance or from background), each edge is labelled
    1 if both spacepoints belong to the same instance, and each labelled
    spacepoint gets the local direction of its instance (y_direction), the
    principal axis of itself and its same-instance neighbours; spacepoints
    with fewer than two such neighbours get a zero vector.

    Args:
        k: Number of nearest neighbours to connect each spacepoint to
        chunk_size: Number of spacepoints to compute distances for at once,
            bounding peak memory to chunk_size * n_sp
    """
    def __init__(self, k: int, chunk_size: int = 1024):
        super().__init__()
        self.k = k
        self.chunk_size = chunk_size

    def forward(self, data: NuGraphData) -> NuGraphData:
        """
        Apply transform to add spacepoint edges and truth labels

        Args:
            data: NuGraph data object to transform
        """

        sp = data["sp"]
        n = sp.num_nodes
        k = min(self.k, n - 1)

        # each chunk of spacepoints is compared against every spacepoint
        if k > 0:
            chunks = []
            for start in range(0, n, self.chunk_size):
                end = min(start + self.chunk_size, n)
                dist = torch.cdist(sp.pos[start:end], sp.pos)
                dist[torch.arange(end - start), torch.arange(start, end)] = torch.inf
                chunks.append(dist.topk(k, largest=False).indices)
            src = torch.cat(chunks, dim=0).reshape(-1)
            dst = torch.arange(n).repeat_interleave(k)
            edge_index = torch.stack((src, dst), dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        edge = data["sp", "sp3d", "sp"]
        edge.edge_index = edge_index

        # truth labels, if true hit instances are available
        if ("hit", "cluster-truth", "particle-truth") in data.edge_types:
            y_i = data.y_i()
            hit, s = data["hit", "nexus", "sp"].edge_index
            lo = torch.full((n,), torch.iinfo(torch.long).max, dtype=torch.long)
            hi = torch.full((n,), -1, dtype=torch.long)
            lo = lo.scatter_reduce(0, s, y_i[hit], reduce="amin")
            hi = hi.scatter_reduce(0, s, y_i[hit], reduce="amax")
            sp.y_instance = torch.where((lo == hi) & (lo >= 0), lo, -1)
            y = sp.y_instance
            edge.y = ((y[edge_index[0]] >= 0) & (y[edge_index[0]] == y[edge_index[1]])).long()

            # local instance direction: principal axis of each spacepoint and
            # its same-instance neighbours
            src, dst = edge_index[:, edge.y == 1]
            pos = torch.cat((sp.pos[src], sp.pos), dim=0).double()
            idx = torch.cat((dst, torch.arange(n)), dim=0)
            count = torch.zeros(n, dtype=pos.dtype).index_add(0, idx, torch.ones_like(idx, dtype=pos.dtype))
            mean = torch.zeros(n, 3, dtype=pos.dtype).index_add(0, idx, pos) / count[:, None]
            d = pos - mean[idx]
            cov = torch.zeros(n, 3, 3, dtype=pos.dtype).index_add(0, idx, d[:, :, None] * d[:, None, :])
            direction = torch.linalg.eigh(cov).eigenvectors[:, :, -1]
            ok = (y >= 0) & (count >= 3)
            sp.y_direction = torch.where(ok[:, None], direction, 0.).float()

        return data
