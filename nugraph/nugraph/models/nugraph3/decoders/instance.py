"""NuGraph3 instance decoder"""
from typing import Any
from cuml import DBSCAN
import torch
from torch import nn
from torchmetrics.functional.clustering import adjusted_rand_score
from torch_geometric.data import Batch
from torch_geometric.utils import cumsum, unbatch
from ....util import ObjCondensationLoss
from ..types import Data, N_IT, E_H_IT, N_IP, E_H_IP

class InstanceDecoder(nn.Module):
    """
    NuGraph3 instance decoder module

    Convolve object condensation node embedding into a beta value and a set of
    coordinates for each hit.

    Args:
        hit_features: Number of hit node features
        instance_features: Number of instance features
    """
    def __init__(self, hit_features: int, instance_features: int):
        super().__init__()

        # loss function
        self.loss = ObjCondensationLoss()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.))

        # network
        self.beta_net = nn.Linear(hit_features, 1)
        self.coord_net = nn.Linear(hit_features, instance_features)

        self.dbscan = DBSCAN()

    # pylint: disable=arguments-differ
    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 instance decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        h = data["hit"]
        device = h.x.device

        # run network and add output to graph object
        h.of = self.beta_net(h.x).squeeze(dim=-1).sigmoid()
        h.ox = self.coord_net(h.x)
        if isinstance(data, Batch):
            # pylint: disable=protected-access
            data._slice_dict["hit"]["of"] = h.ptr
            data._slice_dict["hit"]["ox"] = h.ptr
            data._inc_dict["hit"]["of"] = data._inc_dict["hit"]["x"]
            data._inc_dict["hit"]["ox"] = data._inc_dict["hit"]["x"]

        # calculate loss
        loss = self.loss(h.ox, h.of, data.y_i(), h.y_semantic,
                         data[N_IT].num_nodes, data[E_H_IT].edge_index)
        loss *= (-1 * self.temp).exp()
        b, v = loss
        loss = loss.sum() + self.temp

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"instance/loss-{stage}"] = loss
            metrics[f"instance/bkg-loss-{stage}"] = b
            metrics[f"instance/potential-loss-{stage}"] = v

        # skip materializing instances during training stage
        if stage == "train":
            metrics["temperature/instance"] = self.temp
            return loss, metrics

        # add materialized instances
        mask = torch.ones_like(h.of, dtype=torch.bool)
        if hasattr(h, "x_filter"):
            mask = mask & (h.x_filter > 0.5)
        if hasattr(h, "x_semantic"):
            mask = mask & (h.x_semantic.argmax(dim=1) != 6)
        if isinstance(data, Batch):
            x_ip, e_h_ip = [], []
            for ox, m in zip(unbatch(h.ox, h.batch), unbatch(mask, h.batch)):
                x, e = self.materialize(ox, m)
                x_ip.append(x)
                e_h_ip.append(e)

            # particle nodes
            data[N_IP].x = torch.cat(x_ip, dim=0)
            data[N_IP].batch = torch.cat(
                [torch.full((0,), i, dtype=torch.long, device=device) for i, x in enumerate(x_ip)])
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

            # calculate rand score per graph
            rand = []
            for l in data.to_data_list():
                mask = l["hit"].y_semantic >= 0
                rand.append(adjusted_rand_score(l.x_i()[mask], l.y_i()[mask]))
            rand = torch.stack(rand).mean()

        else:
            data[N_IP].x, data[E_H_IP].edge_index = self.materialize(h.ox, mask)
            rand = adjusted_rand_score(data.x_i(), data.y_i())

        if not -1. < rand < 1.:
            raise RuntimeError(f"Adjusted Rand Score metric value {rand} is outside allowed range!")

        if stage:
            metrics[f"instance/adjusted-rand-{stage}"] = rand

        return loss, metrics

    def materialize(self, ox: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor]:
        """Materialize instance embedding
        
        Args:
            ox: object condensation embedding tensor
            mask: bool mask tensor for background hit removal
        """

        # if there are no signal hits to cluster, skip dbscan and return empty tensors
        if not mask.sum():
            x_ip = torch.empty(0, 0, dtype=torch.float, device=ox.device)
            e_h_ip = torch.empty(2, 0, dtype=torch.long, device=ox.device)
            return x_ip, e_h_ip

        i = torch.empty(ox.size(0), dtype=torch.long, device=ox.device).fill_(-1)
        arr = ox[mask].detach()
        if not arr.is_cuda:
            arr = arr.numpy()
        i[mask] = torch.as_tensor(self.dbscan.fit_predict(arr), dtype=torch.long)
        x_ip = torch.empty(i.max()+1, 0, dtype=torch.float, device=ox.device)
        mask = i > -1
        e_h_ip = torch.stack((torch.nonzero(mask).squeeze(1), i[mask])).long()
        return x_ip, e_h_ip

    def on_epoch_end(self, logger: "WandbLogger", stage: str, epoch: int) -> None:
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: Tensorboard logger object
            stage: Training stage
            epoch: Training epoch index
        """
