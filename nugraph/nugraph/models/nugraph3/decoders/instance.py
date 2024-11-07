"""NuGraph3 instance decoder"""
from typing import Any
import torch
from torch import nn
from torchmetrics.functional.clustering import adjusted_rand_score
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
from pytorch_lightning import LightningModule
from ....util import ObjCondensationLoss
from ..types import Data, N_IT, N_IP

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
                 s_b: float = 0.1, min_degree: int = 1):
        super().__init__()

        self.min_degree = min_degree

        # loss function
        self.loss = ObjCondensationLoss(s_b=s_b)

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.))

        # network
        self.beta_net = nn.Linear(hit_features, 1)
        self.coord_net = nn.Linear(hit_features, instance_features)

    # pylint: disable=arguments-differ
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
            # pylint: disable=protected-access
            data._slice_dict["hit"]["of"] = data["hit"].ptr
            data._slice_dict["hit"]["ox"] = data["hit"].ptr
            data._inc_dict["hit"]["of"] = data._inc_dict["hit"]["x"]
            data._inc_dict["hit"]["ox"] = data._inc_dict["hit"]["x"]

        # calculate loss
        data.materialize_predicted_particles()
        loss = (-1 * self.temp).exp() * self.loss(data, data.y_i()) + self.temp
        b, v = loss
        loss = loss.sum()

        # calculate rand score per graph
        if isinstance(data, Batch):
            rand = torch.mean(torch.stack([adjusted_rand_score(l.x_i(), l.y_i())
                                           for l in data.to_data_list()]))
        else:
            rand = adjusted_rand_score(data.x_i(), data.y_i())
        if not -1. < rand < 1.:
            raise RuntimeError(f"Adjusted Rand Score metric value {rand} is outside allowed range!")

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
                [t.size(0) for t in unbatch(data[N_IP].x, data[N_IP].batch,
                                            batch_size=data.num_graphs)],
                dtype=torch.float)
            metrics[f"instance/num-pred-{stage}"] = num_pred.mean()
            metrics[f"instance/num-true-{stage}"] = num_true.mean()
            metrics[f"instance/num-ratio-{stage}"] = (num_pred/num_true).mean()

            metrics[f"instance/adjusted-rand-{stage}"] = rand

        if stage == "train":
            metrics["temperature/instance"] = self.temp

        return loss, metrics

    def on_epoch_end(self, logger: "WandbLogger", stage: str, epoch: int) -> None:
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: Tensorboard logger object
            stage: Training stage
            epoch: Training epoch index
        """
