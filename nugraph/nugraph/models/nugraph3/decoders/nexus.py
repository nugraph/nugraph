"""NuGraph3 nexus decoder"""
from typing import Any
import torch
from torch import nn
import torchmetrics as tm
from torch_geometric.data import Batch
from pytorch_lightning.loggers import Logger
from ..types import Data

E_SP = ("sp", "sp3d", "sp")

class NexusDecoder(nn.Module):
    """
    NuGraph3 nexus decoder module

    Classify each spacepoint as belonging to a single true particle or not
    (ghost, ambiguous or background spacepoints), and each spacepoint graph
    edge as joining two spacepoints from the same true particle or not.

    Args:
        nexus_features: Number of nexus node features
        vertex: Give the edge classifier each edge's geometry relative to the
            vertex: "pred" for the vertex decoder's prediction, "true" for the
            true vertex (diagnostic only, as it reads truth at inference)
    """
    def __init__(self, nexus_features: int, vertex: str = None):
        super().__init__()

        if vertex not in (None, "pred", "true"):
            raise ValueError(f'"{vertex}" is not a valid nexus vertex option.')
        self.vertex = vertex

        # loss function
        self.loss = nn.BCEWithLogitsLoss()

        # temperature parameters
        self.temp_node = nn.Parameter(torch.tensor(0.))
        self.temp_edge = nn.Parameter(torch.tensor(0.))

        # metrics
        metric_args = {"task": "binary"}
        self.node_recall = tm.Recall(**metric_args)
        self.node_precision = tm.Precision(**metric_args)
        self.edge_recall = tm.Recall(**metric_args)
        self.edge_precision = tm.Precision(**metric_args)

        # network
        self.node_net = nn.Linear(nexus_features, 1)
        self.edge_net = nn.Sequential(
            nn.Linear(2 * nexus_features + 4 + (3 if vertex else 0), nexus_features),
            nn.Mish(),
            nn.Linear(nexus_features, 1))

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 nexus decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        sp, edge = data["sp"], data[E_SP]
        i, j = edge.edge_index

        # run network
        x_node = self.node_net(sp.x).squeeze(dim=-1)
        dpos = sp.pos[i] - sp.pos[j]
        feats = [sp.x[i], sp.x[j], dpos, dpos.norm(dim=1, keepdim=True)]
        if self.vertex:
            feats += self.vertex_features(data, i, j, dpos)
        x_edge = self.edge_net(torch.cat(feats, dim=1)).squeeze(dim=-1)

        # calculate loss
        y_node = (sp.y_instance >= 0).float()
        y_edge = edge.y.float()
        w_node = 2 * (-1 * self.temp_node).exp()
        w_edge = 2 * (-1 * self.temp_edge).exp()
        loss_node = w_node * self.loss(x_node, y_node) + self.temp_node
        loss_edge = w_edge * self.loss(x_edge, y_edge) + self.temp_edge
        loss = loss_node + loss_edge

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"nexus/node-loss-{stage}"] = loss_node
            metrics[f"nexus/edge-loss-{stage}"] = loss_edge
            metrics[f"nexus/node-recall-{stage}"] = self.node_recall(x_node, y_node)
            metrics[f"nexus/node-precision-{stage}"] = self.node_precision(x_node, y_node)
            metrics[f"nexus/edge-recall-{stage}"] = self.edge_recall(x_edge, y_edge)
            metrics[f"nexus/edge-precision-{stage}"] = self.edge_precision(x_edge, y_edge)
        if stage == "train":
            metrics["temperature/nexus-node"] = self.temp_node
            metrics["temperature/nexus-edge"] = self.temp_edge

        # add inference output to graph object
        sp.x_filter = x_node.sigmoid()
        edge.x = x_edge.sigmoid()
        if isinstance(data, Batch):
            # pylint: disable=protected-access
            data._slice_dict["sp"]["x_filter"] = sp.ptr
            data._inc_dict["sp"]["x_filter"] = torch.zeros(data.num_graphs, device=sp.x.device)
            data._slice_dict[E_SP]["x"] = data._slice_dict[E_SP]["edge_index"]
            data._inc_dict[E_SP]["x"] = torch.zeros(data.num_graphs, device=sp.x.device)

        return loss, metrics

    def vertex_features(self, data: Data, i: torch.Tensor, j: torch.Tensor,
                        dpos: torch.Tensor) -> list[torch.Tensor]:
        """
        Edge geometry relative to the vertex: log distance of each spacepoint
        from the vertex, and |cos| of the angle between the edge and the
        radial direction from the vertex

        Args:
            data: Graph data object
            i: Edge target spacepoint indices
            j: Edge source spacepoint indices
            dpos: Edge displacement vectors
        """
        sp = data["sp"]
        vtx = data["evt"].v.detach() if self.vertex == "pred" else data["evt"].y_vtx
        vtx = vtx.reshape(-1, 3).to(sp.pos.dtype)
        batch = sp.batch if isinstance(data, Batch) else torch.zeros_like(sp.pos[:, 0], dtype=torch.long)
        rel = sp.pos - vtx[batch]
        r = rel.norm(dim=1, keepdim=True)
        mid = 0.5 * (rel[i] + rel[j])
        cos = (dpos * mid).sum(dim=1, keepdim=True) / (dpos.norm(dim=1, keepdim=True)
                                                       * mid.norm(dim=1, keepdim=True)).clamp(min=1e-6)
        return [r[i].log1p(), r[j].log1p(), cos.abs()]

    def on_epoch_end(self, logger: Logger | list[Logger], stage: str,
                     epoch: int) -> None: # pylint: disable=unused-argument
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: PyTorch Lightning logger object(s)
            stage: Training stage
            epoch: Training epoch index
        """
