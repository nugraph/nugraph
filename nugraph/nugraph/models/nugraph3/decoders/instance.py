"""NuGraph3 instance decoder"""
from typing import Any
import pathlib
import torch
from torch import nn
from torch_geometric.data import Batch
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
from .base import DecoderBase
from ....util import ObjCondensationLoss
from ..types import T, TD, TDD

class InstanceDecoder(DecoderBase):
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
        super().__init__("instance",
                         planes,
                         None,
                         ObjCondensationLoss(),
                         weight=1.)
        self.dfs = []
        self.planes = planes
        self.net = nn.Linear(planar_features, instance_features+1)

    def forward(self, x: TD) -> TDD:
        """
        NuGraph3 instance decoder forward pass

        Args:
            x: Node embedding tensor dictionary
        """
        x = {p: self.net(x[p]) for p in self.planes}
        return {
            "of": {p: t[:, 0].sigmoid() for p, t in x.items()},
            "ox": {p: t[:, 1:] for p, t in x.items()},
        }

    def loss(self,
             batch: "pyg.HeteroData",
             stage: str,
             confusion: bool = False):
        """
        Calculate loss for NuGraph3 instance decoder

        Args:
            batch: Batch of graph objects
            stage: Training stage
            confusion: Whether to produce confusion matrices
        """
        x, y = self.arrange(batch)
        w = self.weight * (-1 * self.temp).exp()
        loss = w * self.loss_func(x, y) + self.temp
        metrics = {}
        if stage:
            metrics = self.metrics(x, y, stage)
            metrics[f"loss_{self.name}/{stage}"] = loss
            if stage == "train":
                metrics[f"temperature/{self.name}"] = self.temp
        if stage == "val":
            for data in batch.to_data_list():
                if len(self.dfs) >= 100:
                    break
                self.dfs.append(self.draw_event_display(data))

        return loss, metrics

    def arrange(self, batch: TD) -> tuple[T, T]:
        """
        NuGraph3 instance decoder arrange function

        Args:
            batch: Batch of graph objects
        """
        x_coords = torch.cat([batch[p]["ox"] for p in self.planes], dim=0)
        x_filter = torch.cat([batch[p]["of"] for p in self.planes], dim=0)
        y = torch.cat([batch[p]["y_instance"] for p in self.planes], dim=0)
        return (x_coords, x_filter), y

    def materialize(self, data: Batch) -> None:
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
    
    def metrics(self, x: T, y: T, stage: str) -> dict[str, Any]:
        """
        NuGraph3 instance decoder metrics function

        Args:
            x: Model output
            y: Ground truth
            stage: Training stage
        """
        return {}

    # def finalize(self, data) -> None:
    #     """
    #     Finalize outputs for NuGraph3 instance decoder

    #     Args:
    #         data: Graph data object
    #     """
    #     if isinstance(data, Batch):
    #         dlist = data.to_data_list()
    #         for d in dlist:
    #             self.materialize(d)
    #         for p in self.planes:
    #             data[p].i = torch.cat([d[p].i for d in dlist], dim=0)
    #             data._slice_dict[p]["i"] = data._slice_dict[p]["x"]
    #             data._inc_dict[p]["i"] = data._inc_dict[p]["x"]
    #     else:
    #         self.materialize(data)

    def draw_event_display(self, data: "pyg.HeteroData"):
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
