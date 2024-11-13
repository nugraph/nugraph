"""NuGraph3 instance decoder"""
from typing import Any
import cupy as cp
from cuml import DBSCAN
from cuml.internals.memory_utils import using_output_type
from cuml.common.device_selection import using_device_type
import torch
from torch import nn
from torchmetrics.functional.clustering import adjusted_rand_score
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
from pytorch_lightning import LightningModule
from pynuml.data import NuGraphData
from ....util import ObjCondensationLoss
from ..types import Data, N_IT, N_IP, E_H_IP

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

        self.dbscan = DBSCAN()

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

        # add materialized instances
        mask = (data["hit"].x_filter > 0.5) & (data["hit"].x_semantic.argmax(dim=1) != 6)
        if isinstance(data, Batch):
            raise NotImplementedError("Materializing instances for batched graphs in development...")
        else:
            data[N_IP].x, data[E_H_IP].edge_index = self.materialize(data["hit"].ox, mask)

        # calculate loss
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

    def materialize(self, ox: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor]:
        """Materialize instance embedding
        
        Args:
            ox: object condensation embedding tensor
            mask: bool mask tensor for background hit removal
        """
        i = torch.empty(ox.size(0), dtype=torch.long, device=self.device).fill_(-1)
        arr = ox[mask]
        output_type = "cupy" if arr.is_cuda else "numpy"
        arr = cp.from_dlpack(arr) if arr.is_cuda else arr.numpy()
        with using_output_type(output_type):
            arr = self.dbscan.fit_predict(arr)
            i[mask] = torch.from_dlpack(arr).long()
        n_ip = torch.empty(i.max()+1, 0, device=self.device, dtype=torch.float)
        mask = i > -1
        e_h_ip = torch.stack((torch.nonzero(mask).squeeze(1), i[mask])).long()
        return n_ip, e_h_ip

    def on_epoch_end(self, logger: "WandbLogger", stage: str, epoch: int) -> None:
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: Tensorboard logger object
            stage: Training stage
            epoch: Training epoch index
        """
