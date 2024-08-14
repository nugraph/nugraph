"""NuGraph3 instance decoder"""
from typing import Any
import pathlib
import torch
from torch import nn
from torchmetrics.clustering import AdjustedRandScore
from torch_scatter import scatter_min
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
        hit_features: Number of hit node features
        instance_features: Number of instance features
    """
    def __init__(self, hit_features: int, instance_features: int):
        super().__init__()

        # loss function
        self.loss = ObjCondensationLoss()

        # Adjusted Rand Index metric
        self.rand = AdjustedRandScore()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.))

        # network
        self.beta_net = nn.Linear(hit_features, 1)
        self.coord_net = nn.Linear(hit_features, instance_features)

        self.dfs = []

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
            inc = torch.zeros(data.num_graphs, device=data["hit"].x.device)
            data._inc_dict["hit"]["of"] = inc
            data._inc_dict["hit"]["ox"] = inc

        # materialize instances
        materialize = (data[p].of > 0.1).sum() < 2000
        if materialize:
            if isinstance(data, Batch):
                data = Batch([self.materialize(b) for b in data.to_data_list()])
            else:
                self.instance_decoder.materialize(data)

        # calculate loss
        y = data["hit"].y_instance
        w = (-1 * self.temp).exp()
        loss = w * self.loss((data["hit"].ox, data["hit"].of), y) + self.temp

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"loss_instance/{stage}"] = loss
            metrics[f"num_instances/{stage}"] = (data["hit"].of>0.1).sum().float()
        if stage == "train":
            metrics["temperature/instance"] = self.temp

        # disable event displays
        return loss, metrics

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
        device = data["hit"].x.device
        centers = (data["hit"].of > 0.1).nonzero().squeeze(1)
        data["particles"].num_nodes = centers.size(0)
        print(f"generating {centers.size(0)} clusters")
        e = data["hit", "cluster", "particles"]
        e.edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        e.distance = torch.empty(0, dtype=torch.long, device=device)
        for i, center in enumerate(centers):
            print(f"    instance {i+1}")
            center_coords = data["hit"].ox[center]
            dist = (data["hit"].ox - center_coords).square().sum(dim=1).sqrt()
            hits = (dist < 1).nonzero().squeeze(1)
            edge_index = torch.empty(2, hits.size(0), dtype=int, device=device)
            edge_index[0] = hits
            edge_index[1] = i
            e.edge_index = torch.cat((e.edge_index, edge_index), dim=1)
            e.distance = torch.cat((e.distance, dist[hits]), dim=0)

        print("graph has", data["hit", "cluster", "particles"].num_edges, "instance edges")
        
        _, instances = scatter_min(e.distance, e.edge_index[0], dim_size=data["hit"].num_nodes)
        mask = instances != -1
        instances[mask] = e.edge_index[1,instances[mask]]
        data["hit"].i = instances

    def draw_event_display(self, data: HeteroData) -> pd.DataFrame:
        """
        Draw event displays for NuGraph3 object condensation embedding

        Args:
            data: Graph data object
        """
        coords = data["hit"].ox.cpu()
        pca = PCA(n_components=2)
        c1, c2 = pca.fit_transform(coords).transpose()
        beta = data["hit"].of.cpu()
        logbeta = beta.log10()
        xy = data["hit"].pos.cpu()
        i = data["hit"].y_instance.cpu()
        plane = data["hit"].plane.cpu()
        plane = data["hit"].plane.cpu()
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
