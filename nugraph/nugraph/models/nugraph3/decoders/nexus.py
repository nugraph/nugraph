"""NuGraph3 nexus decoder"""
from typing import Any
import math
import torch
from torch import nn
from torch_geometric.utils import scatter, softmax
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
            vertex: "pred" for the vertex decoder's prediction, "vote" for each
            spacepoint's own regressed offset to the vertex, "true" for the
            true vertex (diagnostic only, as it reads truth at inference)
        direction: Regress each spacepoint's local particle direction and give
            the edge classifier the alignment of directions along each edge
        interaction_features: Number of interaction node features, which the
            vertex vote head sees for event-level context
        charge: Give the edge classifier how the hits' charge and width
            change across each edge (needs data["sp"].calo)
        knn_shape: Give the direction head the shape of each spacepoint's
            k-nearest-neighbour neighbourhood (data["sp"].local_shape) rather
            than of its message-passing graph neighbours
        edge_pos_weight: Weight of same-particle edges in the edge loss
        semantic_classes: Give the edge classifier the semantic class
            probabilities of the hits at both ends, from the semantic
            decoder, for this many classes (0 to disable)
        balance_filter: Weight each real spacepoint's ghost-filter loss by
            the inverse square root of the batch frequency of its true
            semantic class (normalised to mean 1, at most 10), so that rare
            classes such as Michel electrons count more (needs
            data["sp"].y_semantic)
        log_length: Give the edge classifier each edge's log length
        stub_count: Give the edge classifier the larger and smaller number
            of Hough stubs at the edge's two ends (needs data["sp"].stub_count)
        edge_layers: Number of hidden layers in the edge classifier
        edge_width: Width of the edge classifier's hidden layers (0 for
            nexus_features)
        vertex_refine: Find the vertex from the spacepoints' votes and
            direction lines (a fixed fit, without gradients), then refine it
            with attention over the spacepoints near it, trained against the
            true vertex (needs vertex "vote" and direction)
    """
    def __init__(self, nexus_features: int, vertex: str = None,
                 direction: bool = False, interaction_features: int = 32,
                 charge: bool = False,
                 knn_shape: bool = False, edge_pos_weight: float = 1.,
                 semantic_classes: int = 0, balance_filter: bool = False,
                 log_length: bool = False, stub_count: bool = False,
                 edge_layers: int = 1, edge_width: int = 0,
                 vertex_refine: bool = False):
        super().__init__()
        if vertex_refine and not (vertex == "vote" and direction):
            raise ValueError("nexus vertex refinement needs vertex votes and directions")

        if vertex not in (None, "pred", "vote", "true"):
            raise ValueError(f'"{vertex}" is not a valid nexus vertex option.')
        self.vertex = vertex
        self.direction = direction
        self.vertex_refine = vertex_refine
        self.charge = charge
        self.knn_shape = knn_shape
        self.semantic_classes = semantic_classes
        self.balance_filter = balance_filter
        self.log_length = log_length
        self.stub_count = stub_count

        # loss function
        self.loss = nn.BCEWithLogitsLoss()
        self.edge_loss = self.loss
        if edge_pos_weight != 1.:
            self.edge_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(edge_pos_weight))

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
        edge_in = (2 * nexus_features + 4 + (3 if vertex else 0) + (3 if direction else 0)
                   + (3 if charge else 0) + 2 * semantic_classes + (1 if log_length else 0)
                   + (2 if stub_count else 0))
        edge_width = edge_width or nexus_features
        hidden = []
        for k in range(edge_layers):
            hidden += [nn.Linear(edge_in if k == 0 else edge_width, edge_width), nn.Mish()]
        self.edge_net = nn.Sequential(*hidden, nn.Linear(edge_width, 1))

        # spacepoint offset to the vertex, in cm
        if vertex == "vote":
            self.temp_vote = nn.Parameter(torch.tensor(0.))
            self.vote_net = nn.Sequential(
                nn.Linear(nexus_features + interaction_features, nexus_features),
                nn.Mish(),
                nn.Linear(nexus_features, 3))

        # spacepoint local particle direction
        if direction:
            self.temp_direction = nn.Parameter(torch.tensor(0.))
            self.direction_net = nn.Sequential(
                nn.Linear(nexus_features + 6, nexus_features),
                nn.Mish(),
                nn.Linear(nexus_features, 3))

        # vertex refinement: an attention weight and a correction (in units of
        # 5 cm) from each spacepoint near the fitted vertex; zero at the start,
        # so the refined vertex begins as the fit
        if vertex_refine:
            self.temp_refine = nn.Parameter(torch.tensor(0.))
            self.refine_net = nn.Sequential(
                nn.Linear(nexus_features + 10, nexus_features),
                nn.Mish(),
                nn.Linear(nexus_features, 4))
            nn.init.zeros_(self.refine_net[-1].weight)
            nn.init.zeros_(self.refine_net[-1].bias)

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
        metrics, extra_loss = {}, 0.
        if self.vertex == "vote":
            evt = data["evt"].x[self.graph_index(data)]
            sp.offset = 100. * self.vote_net(torch.cat((sp.x, evt), dim=1))
            extra_loss = extra_loss + self.vote_loss(data, metrics, stage)
        if self.direction:
            shape = sp.local_shape if self.knn_shape else self.local_shape(sp.pos, i, j)
            sp.direction = nn.functional.normalize(
                self.direction_net(torch.cat((sp.x, shape), dim=1)), dim=1)
            extra_loss = extra_loss + self.direction_loss(data, metrics, stage)
        if self.vertex_refine:
            extra_loss = extra_loss + self.refine_vertex(data, x_node, metrics, stage)
        dpos = sp.pos[i] - sp.pos[j]
        feats = [sp.x[i], sp.x[j], dpos, dpos.norm(dim=1, keepdim=True)]
        if self.vertex:
            feats += self.vertex_features(data, i, j, dpos)
        if self.direction:
            d = sp.direction.detach()
            dn = dpos / dpos.norm(dim=1, keepdim=True).clamp(min=1e-6)
            feats += [(d[i] * d[j]).sum(dim=1, keepdim=True).abs(),
                      (d[i] * dn).sum(dim=1, keepdim=True).abs(),
                      (d[j] * dn).sum(dim=1, keepdim=True).abs()]
        if self.semantic_classes:
            # mean semantic class probabilities of each spacepoint's hits
            hit, s = data["hit", "nexus", "sp"].edge_index
            prob = torch.zeros(sp.num_nodes, self.semantic_classes, device=sp.x.device).index_add(
                0, s, data["hit"].x_semantic[hit].detach())
            prob = prob / prob.sum(dim=1, keepdim=True).clamp(min=1e-6)
            feats += [prob[i], prob[j]]
        if self.charge:
            # change in log charge and log width across the edge, and the
            # larger of the widest hits at its two ends
            calo = sp.calo
            feats += [(calo[i, :1] - calo[j, :1]).abs(),
                      (calo[i, 1:2].clamp(min=1e-3).log() - calo[j, 1:2].clamp(min=1e-3).log()).abs(),
                      torch.maximum(calo[i, 2:], calo[j, 2:])]
        if self.log_length:
            feats.append(dpos.norm(dim=1, keepdim=True).clamp(min=1e-3).log())
        if self.stub_count:
            # larger and smaller number of Hough stubs at the two ends
            count = sp.stub_count[:, None]
            feats += [torch.maximum(count[i], count[j]), torch.minimum(count[i], count[j])]
        x_edge = self.edge_net(torch.cat(feats, dim=1)).squeeze(dim=-1)

        # calculate loss
        y_node = (sp.y_instance >= 0).float()
        y_edge = edge.y.float()
        w_node = 2 * (-1 * self.temp_node).exp()
        w_edge = 2 * (-1 * self.temp_edge).exp()
        if self.balance_filter:
            real = sp.y_semantic >= 0
            cls = sp.y_semantic.clamp(min=0)
            n_cls = self.semantic_classes or int(sp.y_semantic.max()) + 1
            freq = torch.zeros(n_cls, device=cls.device).index_add_(0, cls, real.float())
            w = freq[cls].clamp(min=1).rsqrt()
            w = w / ((w * real).sum() / real.sum().clamp(min=1)).clamp(min=1e-6)
            w = torch.where(real, w.clamp(max=10.), 1.)
            loss_node = w_node * nn.functional.binary_cross_entropy_with_logits(
                x_node, y_node, weight=w) + self.temp_node
        else:
            loss_node = w_node * self.loss(x_node, y_node) + self.temp_node
        loss_edge = self.edge_loss(x_edge, y_edge)
        loss_edge = w_edge * loss_edge + self.temp_edge
        loss = loss_node + loss_edge + extra_loss

        # calculate metrics
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
            for attr in ("offset", "direction"):
                if attr in sp:
                    data._slice_dict["sp"][attr] = sp.ptr
                    data._inc_dict["sp"][attr] = torch.zeros(data.num_graphs, device=sp.x.device)
            for attr in ("v_fit", "v_refine"):
                if attr in data["evt"]:
                    data._slice_dict["evt"][attr] = torch.arange(data.num_graphs + 1, device=sp.x.device)
                    data._inc_dict["evt"][attr] = torch.zeros(data.num_graphs, device=sp.x.device)

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
        if self.vertex == "vote":
            rel = -sp.offset.detach()
        else:
            vtx = data["evt"].v.detach() if self.vertex == "pred" else data["evt"].y_vtx
            rel = sp.pos - self.per_spacepoint(data, vtx)
        r = rel.norm(dim=1, keepdim=True)
        mid = 0.5 * (rel[i] + rel[j])
        cos = (dpos * mid).sum(dim=1, keepdim=True) / (dpos.norm(dim=1, keepdim=True)
                                                       * mid.norm(dim=1, keepdim=True)).clamp(min=1e-6)
        return [r[i].log1p(), r[j].log1p(), cos.abs()]

    @staticmethod
    def graph_index(data: Data) -> torch.Tensor:
        """Graph index of every spacepoint (all zero for a single graph)"""
        sp = data["sp"]
        return sp.batch if isinstance(data, Batch) else torch.zeros_like(sp.pos[:, 0], dtype=torch.long)

    @staticmethod
    def local_shape(pos: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """
        Shape of each spacepoint's neighbourhood: the six independent entries
        of the covariance of its neighbours' offsets, divided by its trace

        Args:
            pos: Spacepoint positions
            i: Edge target spacepoint indices
            j: Edge source spacepoint indices
        """
        d = pos[j] - pos[i]
        cov = torch.zeros(pos.size(0), 3, 3, dtype=pos.dtype, device=pos.device).index_add(
            0, i, d[:, :, None] * d[:, None, :])
        cov = cov / cov.diagonal(dim1=1, dim2=2).sum(dim=1).clamp(min=1e-6)[:, None, None]
        r, c = torch.triu_indices(3, 3, device=pos.device)
        return cov[:, r, c]

    @staticmethod
    def per_spacepoint(data: Data, vtx: torch.Tensor) -> torch.Tensor:
        """
        Broadcast one vertex per graph to every spacepoint in that graph

        Args:
            data: Graph data object
            vtx: Vertex positions, one per graph
        """
        sp = data["sp"]
        vtx = vtx.reshape(-1, 3).to(sp.pos.dtype)
        batch = sp.batch if isinstance(data, Batch) else torch.zeros_like(sp.pos[:, 0], dtype=torch.long)
        return vtx[batch]

    def vote_loss(self, data: Data, metrics: dict, stage: str) -> torch.Tensor:
        """
        Loss on each spacepoint's offset to the true vertex, weighted towards
        spacepoints near the vertex, and the resolution of the vertex
        found by combining the spacepoints' votes

        Args:
            data: Graph data object
            metrics: Metrics dictionary to add to
            stage: Stage name (train/val/test)
        """
        sp = data["sp"]
        target = self.per_spacepoint(data, data["evt"].y_vtx) - sp.pos
        w = (-target.norm(dim=1) / 50.).exp()
        err = (sp.offset - target).norm(dim=1) / 10.
        logcosh = err + nn.functional.softplus(-2. * err) - math.log(2.)
        loss = (w * logcosh).sum() / w.sum().clamp(min=1e-6)
        loss = (-1 * self.temp_vote).exp() * loss + self.temp_vote
        if stage:
            batch = sp.batch if isinstance(data, Batch) else torch.zeros_like(w, dtype=torch.long)
            w_pred = (-sp.offset.detach().norm(dim=1) / 50.).exp()
            vote = scatter(w_pred[:, None] * (sp.pos + sp.offset.detach()), batch, dim=0, reduce="sum")
            vote = vote / scatter(w_pred, batch, dim=0, reduce="sum")[:, None].clamp(min=1e-6)
            res = (vote - data["evt"].y_vtx.reshape(-1, 3)).norm(dim=1)
            metrics[f"nexus/vote-loss-{stage}"] = loss
            metrics[f"nexus/vote-resolution-{stage}"] = res.mean()
            metrics[f"nexus/vote-resolution-median-{stage}"] = res.median()
        return loss

    @staticmethod
    def vote_fit(pos: torch.Tensor, votes: torch.Tensor, dirs: torch.Tensor, w: torch.Tensor,
                 g: torch.Tensor, n: int, scale: float = 2., vote_sigma: float = 5.) -> torch.Tensor:
        """
        Vertex of each graph from its spacepoints' votes and direction lines:
        the mode of the votes (Gaussian mean shift with a 10 cm, then a 3 cm
        kernel, from their weighted mean), then the point closest to the
        direction lines and votes of spacepoints within a shrinking radius of
        it, by iteratively reweighted least squares with Cauchy weights.
        Lines have resolution scale and votes vote_sigma. A batched form of
        the fit in studies/vote_resolution.py, without per-cluster weights

        Args:
            pos: Spacepoint positions
            votes: Spacepoint vertex votes
            dirs: Spacepoint unit directions
            w: Spacepoint vote weights (0 to leave a spacepoint out)
            g: Graph index of each spacepoint
            n: Number of graphs
            scale: Line resolution, cm
            vote_sigma: Vote resolution, cm
        """
        def total(x: torch.Tensor) -> torch.Tensor:
            return scatter(x, g, dim=0, dim_size=n, reduce="sum")

        v = total(w[:, None] * votes) / total(w).clamp(min=1e-6)[:, None]
        for sigma in (10., 3.):
            for _ in range(10):
                k = w * (-0.5 * (votes - v[g]).square().sum(dim=1) / sigma ** 2).exp()
                ks = total(k)
                v = torch.where(ks[:, None] > 1e-12, total(k[:, None] * votes) / ks.clamp(min=1e-12)[:, None], v)
        eye = torch.eye(3, device=pos.device, dtype=torch.float64)
        proj = eye - dirs[:, :, None].double() * dirs[:, None, :].double()
        pos, votes, use = pos.double(), votes.double(), w > 0
        v = v.double()
        for r in (50., 30., 20., 20., 20.):
            near = use & ((pos - v[g]).norm(dim=1) < r)
            enough = total(near.double()) >= 2
            for _ in range(5):
                perp = torch.einsum("nij,nj->ni", proj, v[g] - pos)
                wl = near / (1. + perp.square().sum(dim=1) / scale ** 2)
                wv = near * (scale / vote_sigma) ** 2 / (1. + (v[g] - votes).square().sum(dim=1) / vote_sigma ** 2)
                a = total(wl[:, None, None] * proj) + total(wv)[:, None, None] * eye
                b = total(wl[:, None] * torch.einsum("nij,nj->ni", proj, pos)) + total(wv[:, None] * votes)
                ok = enough & (torch.linalg.cond(a) < 1e8)
                a = torch.where(ok[:, None, None], a, eye)
                v = torch.where(ok[:, None], torch.linalg.solve(a, b), v)
        return v.to(dirs.dtype)

    def refine_vertex(self, data: Data, x_node: torch.Tensor, metrics: dict,
                      stage: str, radius: float = 20.) -> torch.Tensor:
        """
        Fit each graph's vertex from the votes and directions of spacepoints
        kept by the ghost filter (all of them if it keeps none; no gradients), refine it with attention over
        all spacepoints within radius of it, and return the loss of the
        refined vertex against the true vertex

        Args:
            data: Graph data object
            x_node: Ghost filter logits
            metrics: Metrics dictionary to add to
            stage: Stage name (train/val/test)
            radius: Neighbourhood of the fitted vertex the refinement sees, cm
        """
        sp, evt = data["sp"], data["evt"]
        g = self.graph_index(data)
        n = data.num_graphs if isinstance(data, Batch) else 1
        offset, dirs, keep = sp.offset.detach(), sp.direction.detach(), x_node.detach().sigmoid()
        votes = sp.pos + offset
        with torch.no_grad():
            # graphs whose filter keeps no spacepoint (early in training) use them all
            kept = keep > 0.5
            any_kept = scatter(kept.float(), g, dim=0, dim_size=n, reduce="sum") > 0
            w = (-offset.norm(dim=1) / 10.).exp() * (kept | ~any_kept[g])
            fit = self.vote_fit(sp.pos, votes, dirs, w, g, n)
        rel = sp.pos - fit[g]
        near = rel.norm(dim=1) < radius
        gn = g[near]
        out = self.refine_net(torch.cat((sp.x[near], rel[near] / radius,
                                         (votes - fit[g])[near] / radius, dirs[near],
                                         keep[near][:, None]), dim=1))
        att = softmax(out[:, 0], gn, num_nodes=n)
        refine = fit + 5. * scatter(att[:, None] * out[:, 1:], gn, dim=0, dim_size=n, reduce="sum")
        evt.v_fit, evt.v_refine = fit, refine
        has = scatter(near.float(), g, dim=0, dim_size=n, reduce="sum") > 0
        true = evt.y_vtx.reshape(-1, 3)
        err = (refine - true).norm(dim=1)[has] / 2.
        logcosh = err + nn.functional.softplus(-2. * err) - math.log(2.)
        loss = logcosh.mean() if has.any() else refine.sum() * 0.
        loss = (-1 * self.temp_refine).exp() * loss + self.temp_refine
        if stage:
            res_fit = (fit - true).norm(dim=1)
            res_refine = (refine.detach() - true).norm(dim=1)
            metrics[f"nexus/refine-loss-{stage}"] = loss
            metrics[f"nexus/fit-resolution-median-{stage}"] = res_fit.median()
            metrics[f"nexus/refine-resolution-median-{stage}"] = res_refine.median()
            metrics[f"nexus/refine-within-3cm-{stage}"] = (res_refine < 3.).float().mean()
            metrics[f"nexus/fit-within-3cm-{stage}"] = (res_fit < 3.).float().mean()
        if stage == "train":
            metrics["temperature/nexus-refine"] = self.temp_refine
        return loss

    def direction_loss(self, data: Data, metrics: dict, stage: str) -> torch.Tensor:
        """
        Sign-invariant loss on each spacepoint's local particle direction

        Args:
            data: Graph data object
            metrics: Metrics dictionary to add to
            stage: Stage name (train/val/test)
        """
        sp = data["sp"]
        # spacepoints without a true direction are left out by weight, since
        # selecting them makes the CPU wait on the GPU
        m = (sp.y_direction.norm(dim=1) > 0).float()
        n = m.sum().clamp(min=1)
        cos = (sp.direction * sp.y_direction).sum(dim=1).abs()
        loss = ((1. - cos) * m).sum() / n
        loss = (-1 * self.temp_direction).exp() * loss + self.temp_direction
        if stage:
            metrics[f"nexus/direction-loss-{stage}"] = loss
            metrics[f"nexus/direction-cos-{stage}"] = (cos * m).sum() / n
        return loss

    def on_epoch_end(self, logger: Logger | list[Logger], stage: str,
                     epoch: int) -> None: # pylint: disable=unused-argument
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: PyTorch Lightning logger object(s)
            stage: Training stage
            epoch: Training epoch index
        """
