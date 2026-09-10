"""NuGraph3 data transform"""
import torch
from torch_geometric.transforms import BaseTransform
from pynuml.data import NuGraphData

class Transform(BaseTransform):
    """
    NuGraph3 data transform
    
    Args:
        planes: Tuple of detector plane names
    """
    def __init__(self, planes: tuple[str]):
        super().__init__()
        self.planes = planes

    def forward(self, data: NuGraphData) -> NuGraphData:

        """
        Apply transform for compatibility with NuGraph3 model

        Args:
           data: NuGraph data object to transform
        """

        # transform old planar format into new hierarchical format
        if "hit" not in data.node_types:

            # unify planar edges
            edge_plane = []
            edge_nexus = []
            for i, p in enumerate(self.planes):
                offset = 0
                for j in range(i): # get offset from previous planes
                    offset += data[self.planes[j]].num_nodes
                edge_plane.append(data[p, "plane", p].edge_index + offset)
                del data[p, "plane", p]
                edge_nexus.append(data[p, "nexus", "sp"].edge_index)
                edge_nexus[-1][0] += offset # increment only the plane node index
                del data[p, "nexus", "sp"]
            data["hit", "delaunay-planar", "hit"].edge_index = torch.cat(edge_plane, dim=1)
            data["hit", "nexus", "sp"].edge_index = torch.cat(edge_nexus, dim=1)

            # add plane index to feature tensor
            for i, p in enumerate(self.planes):
                data[p].plane = torch.empty_like(data[p].x[:,0], dtype=int).fill_(i)
                data[p].x = torch.cat([data[p].x, data[p].plane.unsqueeze(1)], dim=1)

            # merge planar node stores
            for attr in data[self.planes[0]].node_attrs():
                data["hit"][attr] = torch.cat([data[p][attr] for p in self.planes], dim=0)
            for p in self.planes:
                del data[p]

            # add true instance nodes
            if hasattr(data["hit"], "y_instance"):
                y = data["hit"].y_instance
                mask = y != -1
                y = y[mask]
                instances = y.unique()
                # remap instances
                imax = instances.max() + 1 if instances.size(0) else 0
                if instances.size(0) != imax:
                    remap = torch.full((imax,), -1, dtype=torch.long)
                    remap[instances] = torch.arange(instances.size(0))
                    y = remap[y]
                data["particle-truth"].x = torch.empty(instances.size(0), 0)
                edges = torch.stack((mask.nonzero().squeeze(1), y), dim=0).long()
                data["hit", "cluster-truth", "particle-truth"].edge_index = edges
                del data["hit"].y_instance

            # add edges to and from event node
            data["evt"].x = torch.empty((1, 0))
            lo = torch.arange(data["hit"].num_nodes, dtype=torch.long)
            hi = torch.zeros(data["hit"].num_nodes, dtype=torch.long)
            data["hit", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)
            lo = torch.arange(data["sp"].num_nodes, dtype=torch.long)
            hi = torch.zeros(data["sp"].num_nodes, dtype=torch.long)
            data["sp", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)

        # rename true hit position (remove once spacepoint decoder is mature)
        if "c" in data["hit"].keys():
            data["hit"].y_position = data["hit"].c
            del data["hit"].c

        # ensure event truth labels have correct format
        evt = data["evt"]
        if not evt.y.ndim:
            evt.y = evt.y.reshape([1])

        # concatenate position tensor onto node features
        h = data["hit"]
        h.x = torch.cat((h.pos, h.x), dim=-1)

        # construct pmt-pmt edges if not already present in the dataset
        if "pmt" in data.node_types and ("pmt", "knn", "pmt") not in data.edge_types:
            n_pmt = data["pmt"].pos.size(0)
            if n_pmt > 1:
                distances = torch.cdist(data["pmt"].pos, data["pmt"].pos, p=2)
                distances.fill_diagonal_(float("inf"))
                knn = min(3, n_pmt - 1)
                _, neighbor_idx = torch.topk(distances, knn, largest=False, dim=1)
                source = torch.arange(n_pmt, dtype=torch.long).repeat_interleave(knn)
                target = neighbor_idx.flatten()
                edge_pmt = torch.stack((source, target), dim=0)
                edge_pmt = torch.cat((edge_pmt, edge_pmt.flip(0)), dim=1)
            else:
                edge_pmt = torch.empty((2, 0), dtype=torch.long)
            data["pmt", "knn", "pmt"].edge_index = edge_pmt

        # ensure optical edge tensors keep their shape through batching
        for edge_type in [("flash", "in", "evt"), ("ophit", "in", "pmt"),
                          ("pmt", "knn", "pmt"), ("sp", "knn", "pmt")]:
            if edge_type in data.edge_types:
                if data[edge_type].edge_index.dim() == 1:
                    data[edge_type].edge_index = data[edge_type].edge_index.unsqueeze(1)

        return data
