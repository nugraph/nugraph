"""NuGraph3 instance decoder"""
from typing import Any
from sklearn.cluster import DBSCAN
import torch
from torch import nn
from torchmetrics.functional.clustering import adjusted_rand_score
from torch_geometric.data import Batch
from torch_geometric.utils import cumsum, unbatch
from ....util import ObjConLoss, RecallLoss
from ..types import Data, N_IT, E_H_IT, N_IP, E_H_IP

# largest block of pairwise distances computed at once by the seed materializer
MAX_PAIRS = 2 ** 22

def _blocks(n_rows: int, n_cols: int) -> list[slice]:
    """Row slices whose block of pairwise distances has at most MAX_PAIRS elements"""
    step = max(1, MAX_PAIRS // max(1, n_cols))
    return [slice(k, k + step) for k in range(0, n_rows, step)]

class InstanceDecoder(nn.Module):
    """
    NuGraph3 instance decoder module

    Convolve object condensation node embedding into a beta value and a set of
    coordinates for each hit. Particle instances are materialized from these
    either with DBSCAN or from beta-seeded condensation points.

    Args:
        beta_features: Number of object condensation beta features
        coord_features: Number of object condensation coordinate features
        instance_features: Number of instance features
        semantic_classes: List of names of semantic classes
        dbscan_eps: Epsilon hyperparameter for DBSCAN algorithm
        particle_loss: Whether to compute particle loss term
        materializer: Instance materialization method, "dbscan" or "seed"
        seed_beta: Minimum beta for a seed condensation point
        seed_radius: Radius within which a seed suppresses lower-beta seeds
        seed_assign_radius: Maximum distance from a hit to its condensation point
    """
    def __init__(self, beta_features: int, coord_features: int,
                 instance_features: int, semantic_classes: list[str],
                 dbscan_eps: float = 0.3, particle_loss: bool = False,
                 materializer: str = "dbscan", seed_beta: float = 0.5,
                 seed_radius: float = 0.3, seed_assign_radius: float = float("inf")):
        super().__init__()

        # loss function
        self.loss = ObjConLoss()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.))

        # beta decoder
        self.beta_net = nn.Sequential(
            nn.Linear(beta_features + len(semantic_classes), beta_features),
            nn.Mish(),
            nn.Linear(beta_features, 1),
            nn.Sigmoid(),
        )

        # coordinate decoder
        self.coord_net = nn.Sequential(
            nn.Linear(coord_features + len(semantic_classes), coord_features),
            nn.Mish(),
            nn.Linear(coord_features, instance_features),
        )

        self.eps = dbscan_eps
        self.particle_loss = particle_loss

        # hits predicted as diffuse are not clustered into particles
        self.diffuse = (semantic_classes.index("diffuse")
                        if "diffuse" in semantic_classes else None)

        if materializer not in ("dbscan", "seed"):
            raise ValueError(f'"{materializer}" is not a valid materializer.')
        self.materializer = materializer
        self.seed_beta = seed_beta
        self.seed_radius = seed_radius
        self.seed_assign_radius = seed_assign_radius

    # pylint: disable=arguments-differ
    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 instance decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        h = data["hit"]

        # run network and add output to graph object
        h.of = self.beta_net(torch.cat((h.of, h.x_semantic), dim=-1)).squeeze(dim=-1)
        h.ox = self.coord_net(torch.cat((h.ox, h.x_semantic), dim=-1))

        if isinstance(data, Batch):
            # pylint: disable=protected-access
            data._slice_dict["hit"]["of"] = h.ptr
            data._slice_dict["hit"]["ox"] = h.ptr
            data._inc_dict["hit"]["of"] = data._inc_dict["hit"]["x"]
            data._inc_dict["hit"]["ox"] = data._inc_dict["hit"]["x"]

        # calculate semantic loss to input to object condensation particle loss
        loss_semantic = None
        if (self.particle_loss):
            loss_semantic = data.hit_loss()

        # calculate loss
        loss = self.loss(h.ox, h.of, data.y_i(), h.y_semantic,
                         data[N_IT].num_nodes, data[E_H_IT].edge_index,
                         loss_semantic)
        loss *= (-1 * self.temp).exp()
        b, v, p = loss
        loss = loss.sum() + self.temp

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"instance/loss-{stage}"] = loss
            metrics[f"instance/bkg-loss-{stage}"] = b
            metrics[f"instance/potential-loss-{stage}"] = v
            if self.particle_loss:
                metrics[f"instance/particle-loss-{stage}"] = p

        if not self.training:

            self.materialize(data)
            rand = self.adjusted_rand_score(data, masked=False)

            if not -1. <= rand <= 1.:
                raise RuntimeError(f"Adjusted Rand Score metric value {rand} is outside allowed range!")

            if stage:
                metrics[f"instance/adjusted-rand-{stage}"] = rand

        if stage == "train":
            metrics["temperature/instance"] = self.temp

        return loss, metrics

    def materialize(self, data: Data) -> None:
        """Materialize a graph or batch

        Args:
            data: Graph data object
        """

        h = data["hit"]
        device = h.x.device

        mask = torch.ones_like(h.of, dtype=torch.bool)
        if hasattr(h, "x_filter"):
            mask = mask & (h.x_filter > 0.5)
        if hasattr(h, "x_semantic") and self.diffuse is not None:
            mask = mask & (h.x_semantic.argmax(dim=1) != self.diffuse)

        if isinstance(data, Batch):
            x_ip, e_h_ip = [], []
            for ox, of, m in zip(unbatch(h.ox, h.batch), unbatch(h.of, h.batch),
                                 unbatch(mask, h.batch)):
                x, e = self.cluster(ox, of, m)
                x_ip.append(x)
                e_h_ip.append(e)

            # particle nodes
            data[N_IP].x = torch.cat(x_ip, dim=0)
            data[N_IP].batch = torch.cat(
                [torch.full((x.size(0),), i, dtype=torch.long, device=device)
                 for i, x in enumerate(x_ip)])
            data[N_IP].ptr = cumsum(torch.tensor([x.size(0) for x in x_ip], device=device))
            data._slice_dict[N_IP] = {"x": data[N_IP].ptr} # pylint: disable=protected-access
            data._inc_dict[N_IP] = { # pylint: disable=protected-access
                "x": torch.zeros(data.num_graphs, dtype=torch.long, device=device)
            }

            # particle edges
            e_inc = torch.stack((h.ptr[:-1], data[N_IP].ptr[:-1]), dim=1).unsqueeze(2)
            data[E_H_IP].edge_index = torch.cat([e + inc for e, inc in zip(e_h_ip, e_inc)], dim=1)
            data._slice_dict[E_H_IP] = { # pylint: disable=protected-access
                "edge_index": cumsum(torch.tensor([e.size(1) for e in e_h_ip]))
            }
            data._inc_dict[E_H_IP] = {"edge_index": e_inc} # pylint: disable=protected-access

        else:
            data[N_IP].x, data[E_H_IP].edge_index = self.cluster(h.ox, h.of, mask)

    def cluster(self, ox: torch.Tensor, of: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor]:
        """Materialize one graph with the configured method

        Args:
            ox: object condensation embedding tensor
            of: object condensation beta tensor
            mask: bool mask tensor for background hit removal
        """
        if self.materializer == "seed":
            return self.seed(ox, of, mask)
        return self.dbscan(ox, mask)

    def dbscan(self, ox: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor]:
        """Materialize instance embedding using DBSCAN
        
        Args:
            ox: object condensation embedding tensor
            mask: bool mask tensor for background hit removal
        """

        # if there are no signal hits to cluster, skip dbscan and return empty tensors
        if not mask.sum():
            x_ip = torch.empty(0, 0, dtype=ox.dtype, device=ox.device)
            e_h_ip = torch.empty(2, 0, dtype=torch.long, device=ox.device)
            return x_ip, e_h_ip

        i = torch.empty(ox.size(0), dtype=torch.long, device=ox.device).fill_(-1)
        arr = ox[mask].detach().to(torch.float32).cpu().numpy()
        labels = DBSCAN(eps=self.eps).fit_predict(arr)
        i[mask] = torch.from_numpy(labels).to(device=ox.device, dtype=torch.long)
        x_ip = torch.empty(i.max()+1, 0, dtype=ox.dtype, device=ox.device)
        mask = i > -1
        e_h_ip = torch.stack((torch.nonzero(mask).squeeze(1), i[mask])).long()
        return x_ip, e_h_ip

    def seed(self, ox: torch.Tensor, of: torch.Tensor,
             mask: torch.Tensor) -> tuple[torch.Tensor]:
        """Materialize instance embedding from beta-seeded condensation points

        Hits with beta above seed_beta are candidate condensation points. A
        candidate is dropped if a candidate with higher beta lies within
        seed_radius, and each hit is assigned to its nearest remaining
        condensation point if that lies within seed_assign_radius. This runs on
        the embedding's device, without copying to the CPU.

        Args:
            ox: object condensation embedding tensor
            of: object condensation beta tensor
            mask: bool mask tensor for background hit removal
        """
        x = ox[mask].detach().float()
        beta = of[mask].detach().float()
        exact = "donot_use_mm_for_euclid_dist"

        # candidate condensation points; ties in beta are broken by hit index
        cand = torch.nonzero(beta > self.seed_beta).squeeze(1)
        keep = torch.ones_like(cand, dtype=torch.bool)
        for rows in _blocks(cand.size(0), cand.size(0)):
            dist = torch.cdist(x[cand[rows]], x[cand], compute_mode=exact)
            b_row, b_col = beta[cand[rows]].unsqueeze(1), beta[cand].unsqueeze(0)
            higher = (b_col > b_row) | ((b_col == b_row) & (cand < cand[rows].unsqueeze(1)))
            keep[rows] = ~((dist <= self.seed_radius) & higher).any(dim=1)
        points = x[cand[keep]]

        # assign each hit to its nearest condensation point
        labels = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
        if points.size(0):
            for rows in _blocks(x.size(0), points.size(0)):
                dist, nearest = torch.cdist(x[rows], points, compute_mode=exact).min(dim=1)
                labels[rows] = torch.where(dist <= self.seed_assign_radius, nearest, -1)

        i = torch.full((ox.size(0),), -1, dtype=torch.long, device=ox.device)
        i[mask] = labels
        x_ip = torch.empty(points.size(0), 0, dtype=ox.dtype, device=ox.device)
        mask = i > -1
        e_h_ip = torch.stack((torch.nonzero(mask).squeeze(1), i[mask])).long()
        return x_ip, e_h_ip

    def adjusted_rand_score(self, data: Data, masked: bool = False) -> torch.Tensor:
        """Calculate adjusted rand score for batch

        Args:
            data: Graph data object
            masked: Whether to mask background hits
        """
        if isinstance(data, Batch):
            data_list = data.to_data_list()
        else:
            data_list = [data]

        rand = []
        for d in data_list:
            x, y = d.x_i(), d.y_i()
            if masked:
                mask = d["hit"].y_semantic >= 0
                x, y = x[mask], y[mask]
            rand.append(adjusted_rand_score(x, y))
        return torch.stack(rand).mean()

    def on_epoch_end(self, logger: "WandbLogger", stage: str, epoch: int) -> None:
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: Tensorboard logger object
            stage: Training stage
            epoch: Training epoch index
        """
