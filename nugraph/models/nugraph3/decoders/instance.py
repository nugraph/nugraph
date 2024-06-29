"""NuGraph3 instance decoder"""
from typing import Any
import pathlib
import torch
from torch import nn
from torch_geometric.data import Batch, HeteroData
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
from ....util import ObjCondensationLoss
from ..types import Data

class InstanceDecoder(nn.Module):
    """
    NuGraph3 instance decoder module

    Convolve object condensation node embedding into a beta value and a set of
    coordinates for each hit.

    Args:
        planar_features: Number of planar features
        instance_features: Number of instance features
        planes: List of detector planes
    """
    def __init__(self, planar_features: int,
                 instance_features: int, planes: list[str]):
        super().__init__()

        # loss function
        self.loss = ObjCondensationLoss()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.))

        # network
        self.beta_net = nn.Linear(planar_features, 1)
        self.coord_net = nn.Linear(planar_features, instance_features)

        self.dfs = []
        self.planes = planes

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 instance decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        # run network and add output to graph object
        for p in self.planes:
            data[p].of = self.beta_net(data[p].x).squeeze(dim=-1).sigmoid()
            data[p].ox = self.coord_net(data[p].x)
            if isinstance(data, Batch):
                data._slice_dict[p]["of"] = data[p].ptr
                data._slice_dict[p]["ox"] = data[p].ptr
                inc = torch.zeros(data.num_graphs, device=data[p].x.device)
                data._inc_dict[p]["of"] = inc
                data._inc_dict[p]["ox"] = inc

        # calculate loss
        of = torch.cat([data[p].of for p in self.planes], dim=0)
        ox = torch.cat([data[p].ox for p in self.planes], dim=0)
        y = torch.cat([data[p].y_instance for p in self.planes], dim=0)
        w = (-1 * self.temp).exp()
        loss = w * self.loss((ox, of), y) + self.temp

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"loss_instance/{stage}"] = loss
            metrics[f"num_instances/{stage}"] = (of>0.1).sum().float()
        if stage == "train":
            metrics["temperature/instance"] = self.temp
        if stage == "val" and isinstance(data, Batch):
            for d in data.to_data_list():
                if len(self.dfs) >= 100:
                    break
                self.dfs.append(self.draw_event_display(d))

        return loss, metrics

    def materialize(self, data: Data) -> None:
        """
        Materialize object condensation embedding into instances

        Args:
            data: Heterodata graph object
        """
        of = torch.cat([data[p].of for p in self.planes], dim=0)
        ox = torch.cat([data[p].ox for p in self.planes], dim=0)
        centers = (of > 0.1).nonzero().squeeze(0)
        distances = []
        for center in centers:
            center_coords = ox[center]
            dist = 1 - (ox - center_coords).square().sum(dim=1).sqrt()
            dist[(dist<0)] = 0
            distances.append(dist)
        if distances:
            distances = torch.stack(distances, dim=1)
            total = distances.sum(dim=1)
            mask = (total > 0)
            xi = distances.argmax(dim=1)
            xi[~mask] = -1
            ts = xi.split([data[p].num_nodes for p in self.planes])
            for p, t in zip(self.planes, ts):
                data[p].i = t.int()
        else:
            for p in self.planes:
                data[p].i = torch.empty_like(data[p].of).fill_(-1).int()

    def draw_event_display(self, data: HeteroData) -> pd.DataFrame:
        """
        Draw event displays for NuGraph3 object condensation embedding

        Args:
            data: Graph data object
        """
        coords = torch.cat([data[p].ox for p in self.planes], dim=0).cpu()
        pca = PCA(n_components=2)
        c1, c2 = pca.fit_transform(coords).transpose()
        beta = torch.cat([data[p].of for p in self.planes], dim=0).cpu()
        logbeta = beta.log10()
        xy = torch.cat([data[p].pos for p in self.planes], dim=0).cpu()
        i = torch.cat([data[p].y_instance for p in self.planes], dim=0).cpu()
        plane = [p for p in self.planes for _ in range(data[p].num_nodes)]
        return pd.DataFrame(dict(c1=c1, c2=c2, beta=beta, logbeta=logbeta,
                                 plane=plane, x=xy[:,0], y=xy[:,1],
                                 instance=pd.Series(i).astype(str)))

    def on_epoch_end(self,
                     logger: TensorBoardLogger,
                     stage: str,
                     epoch: int) -> None:
        """
        NuGraph3 instance decoder end-of-epoch callback function

        Args:
            logger: Tensorboard logger object
            stage: Training stage
            epoch: Training epoch index
        """
        if not logger:
            return
        path = pathlib.Path(logger.log_dir) / "objcon-plots"
        path.mkdir(exist_ok=True)
        if stage == "val":
            for i, df in enumerate(self.dfs):

                # object condensation true instance plot
                fig = px.scatter(df, x="x", y="y", facet_col="plane",
                                 color="instance", title=f"epoch {epoch}")
                fig.update_xaxes(matches=None)
                for a in fig.layout.annotations:
                    a.text = a.text.replace("plane=", "")
                fig.write_image(file=path/f"evt{i+1}-truth.png")

                # object condensation beta plot
                fig = px.scatter(df, x="x", y="y", facet_col="plane",
                                 color="logbeta", title=f"epoch {epoch}")
                fig.update_xaxes(matches=None)
                for a in fig.layout.annotations:
                    a.text = a.text.replace("plane=", "")
                fig.write_image(file=path/f"evt{i+1}-beta.png")

                # object condensation coordinate plot
                fig = px.scatter(df, x="c1", y="c2",
                                 color="instance", title=f"epoch {epoch}")
                fig.write_image(file=path/f"evt{i+1}-coords.png")

        self.dfs = []
