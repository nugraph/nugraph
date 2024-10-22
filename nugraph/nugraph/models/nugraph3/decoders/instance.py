"""NuGraph3 instance decoder"""
from typing import Any
import torch
from torch import nn
from torchmetrics.clustering import AdjustedRandScore
from torch_scatter import scatter_min
from torch_geometric.data import Batch
from pytorch_lightning import LightningModule
from ....util import ObjCondensationLoss
from ..types import Data, N_IT, N_IP, E_H_IT, E_H_IP

class InstanceDecoder(LightningModule):
    """
    NuGraph3 instance decoder module

    Convolve object condensation node embedding into a beta value and a set of
    coordinates for each hit.

    Args:
        hit_features: Number of hit node features
        instance_features: Number of instance features
        s_b: Background suppression hyperparameter
    """
    def __init__(self, hit_features: int, instance_features: int,
                 s_b: float = 0.1):
        super().__init__()

        # loss function
        self.loss = ObjCondensationLoss(s_b=s_b)

        # Adjusted Rand Index metric
        self.rand = AdjustedRandScore()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.))

        # network
        self.beta_net = nn.Linear(hit_features, 1)
        self.coord_net = nn.Linear(hit_features, instance_features)

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 instance decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        # run network and add output to graph object
        data["hit"].of = self.beta_net(data["hit"].x).squeeze(dim=-1).sigmoid()
        data["hit"].ox = self.coord_net(data["hit"].x)
        if isinstance(data, Batch):
            data._slice_dict["hit"]["of"] = data["hit"].ptr
            data._slice_dict["hit"]["ox"] = data["hit"].ptr
            data._inc_dict["hit"]["of"] = data._inc_dict["hit"]["x"]
            data._inc_dict["hit"]["ox"] = data._inc_dict["hit"]["x"]

        # materialize instances
        materialize = (data["hit"].of > 0.1).sum() < 2000
        if materialize:
            # form instances across batch
            imask = data["hit"].of > 0.1
            data[N_IP].x = torch.empty(imask.sum(), 0, device=self.device)
            data[N_IP].ox = data["hit"].ox[imask]
            if isinstance(data, Batch):
                repeats = torch.empty(data.num_graphs, dtype=torch.long, device=self.device)
                data[N_IP].batch = torch.empty(data[N_IP].num_nodes,
                                               dtype=torch.long, device=self.device)
                for i in range(data.num_graphs):
                    lo, hi = data._slice_dict["hit"]["x"][i:i+2]
                    repeats[i] = imask[lo:hi].sum()
                    data[N_IP].batch[lo:hi] = i
                data[N_IP].ptr = cumsum(repeats)
                data._slice_dict[N_IP] = {
                    "x": data[N_IP].ptr,
                    "ox": data[N_IP].ptr,
                }
                data._inc_dict[N_IP] = {
                    "x": data._inc_dict["hit"]["x"],
                    "ox": data._inc_dict["hit"]["x"],
                }
                data = Batch.from_data_list([self.materialize(b) for b in data.to_data_list()])
            else:
                self.materialize(data)

            # collapse instance edges into labels
            e = data[E_H_IP]
            _, instances = scatter_min(e.distance, e.edge_index[0], dim_size=data["hit"].num_nodes)
            mask = instances < e.num_edges
            instances[~mask] = -1
            instances[mask] = e.edge_index[1, instances[mask]]
            data["hit"].i = instances
            if isinstance(data, Batch):
                data._slice_dict["hit"]["i"] = data["hit"].ptr
                data._inc_dict["hit"]["i"] = data._inc_dict["hit"]["x"]

        # calculate loss
        y = torch.full_like(data["hit"].y_semantic, -1)
        i, j = data[E_H_IT].edge_index
        y[i] = j
        data["hit"].y_instance = y
        loss = (-1 * self.temp).exp() * self.loss(data, y) + self.temp
        b, v = loss
        loss = loss.sum()

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"instance/loss-{stage}"] = loss
            metrics[f"instance/bkg-loss-{stage}"] = b
            metrics[f"instance/potential-loss-{stage}"] = v

            # number of instances
            num_true = torch.tensor(
                [t.size(0) for t in unbatch(data[N_IT].x, data[N_IT].batch,
                                            batch_size=data.num_graphs)],
                dtype=torch.float)
            num_pred = torch.tensor(
                [(t>0.1).sum() for t in unbatch(data["hit"].of, data["hit"].batch,
                                                batch_size=data.num_graphs)],
                dtype=torch.float)
            metrics[f"instance/num-pred-{stage}"] = num_pred.mean()
            metrics[f"instance/num-true-{stage}"] = num_true.mean()
            metrics[f"instance/num-ratio-{stage}"] = (num_pred/num_true).mean()

            if materialize:
                x = data["hit"].i
                metrics[f"instance/adjusted-rand-{stage}"] = self.rand(x, y)

        if stage == "train":
            metrics["temperature/instance"] = self.temp

        return loss, metrics

    def materialize(self, data: Data) -> None:
        """
        Materialize object condensation embedding into instances

        Args:
            data: Heterodata graph object
        """
        e = data[E_H_IP]
        x_hit = data["hit"].ox
        x_part = data[N_IP].ox
        dist = (x_hit[:, None, :] - x_part[None, :, :]).square().sum(dim=2)
        e.edge_index = (dist < 1).nonzero().transpose(0, 1).detach()
        e.distance = dist[e.edge_index[0], e.edge_index[1]].detach()
        return data

    def on_epoch_end(self, logger: "WandbLogger", stage: str, epoch: int) -> None:
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: Tensorboard logger object
            stage: Training stage
            epoch: Training epoch index
        """
