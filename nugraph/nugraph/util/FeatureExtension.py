import torch
from torch_geometric.transforms import BaseTransform
from torch import norm, topk, zeros, cat, stack, log

class FeatureExtension(BaseTransform):

    def __init__(self, planes: list[str]):
        super().__init__()
        self.planes = planes

    def __call__(self, data: "pyg.data.HeteroData") -> "pyg.data.HeteroData":

        for p in self.planes:
            ## Adding delta wire an delta time (dwire/dtime doesn't work; some infs)
            # Extracting wire and time information
            wt_coords = stack((data.collect("pos")[p][:, 0], data.collect("pos")[p][:, 1]), dim=1) # [wire, time]

            # Calculating pairwise euclidean distances of nodes in the wire vs time space
            dist_table = norm(wt_coords[:, None, :] - wt_coords[None, :, :], dim=-1)
            dist_table.fill_diagonal_(float('inf'))

            # Find a (n_nodes, 2) matrix containing the distances and indexes of the two closest nodes to each node
            dists_2closest_nodes, idxs_2closest_nodes = topk(dist_table, 2, dim=1, largest=False, sorted=True)

            # Finding the ratio of the wire and time differences of the two closest neighbors
            #dwire = (wt_coords[idxs_2closest_nodes[:,1], 0] - wt_coords[idxs_2closest_nodes[:,0], 0]).view(-1,1)
            #dtime = (wt_coords[idxs_2closest_nodes[:,1], 1] - wt_coords[idxs_2closest_nodes[:,0], 1]).view(-1,1)
            # Double delta (Giuseppe suggestion)
            dwire = (2*wt_coords[:, 0] - wt_coords[idxs_2closest_nodes[:,1], 0] - wt_coords[idxs_2closest_nodes[:,0], 0]).view(-1,1)
            dtime = (2*wt_coords[:, 1] - wt_coords[idxs_2closest_nodes[:,1], 1] - wt_coords[idxs_2closest_nodes[:,0], 1]).view(-1,1)

            ## Adding node degree
            nodes_degree = torch.unique(data[p, 'plane', p].edge_index[0], sorted=True, return_counts=True)[1].view(-1,1)
            nodes_degree = log(nodes_degree) # Should I use log(nodes_degree) instead?

            ## Adding shortest edge length
            min_dist = dists_2closest_nodes[:,0].view(-1,1) # 'dists_2closest_nodes' is sorted in ascending order

            # Extending the original node feature matrix with the new features
            data[p].x = cat((data[p].x, dwire, dtime, nodes_degree, min_dist), dim=-1)

        return data
