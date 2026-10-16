"""Tests for the NuGraph3 nexus decoder"""
import torch
from torch_geometric.data import Batch

from nugraph.models.nugraph3 import NuGraph3
from nugraph.models.nugraph3.transform import Transform
from nugraph.util import NexusFeatures, SpacePointGraph

from synthetic import EVENT_CLASSES, IN_FEATURES, PLANES, SEMANTIC_CLASSES, make_event

E_SP = ("sp", "sp3d", "sp")


def nexus_batch(*seeds: int, sp_feats: bool = False, charge: bool = False,
                graph: str = "knn") -> Batch:
    """Batch of synthetic events with spacepoint graph and truth labels"""
    transforms = [Transform(PLANES), SpacePointGraph(k=4, graph=graph)]
    if sp_feats:
        transforms.append(NexusFeatures(PLANES, positions=False, charge=charge))
    events = []
    for s in seeds:
        data = make_event(s)
        for t in transforms:
            data = t(data)
        events.append(data)
    return Batch.from_data_list(events)


def nexus_model(**kwargs) -> NuGraph3:
    """NuGraph3 with semantic, filter and nexus heads"""
    torch.manual_seed(0)
    return NuGraph3(in_features=IN_FEATURES, planes=PLANES,
                    semantic_classes=SEMANTIC_CLASSES, event_classes=EVENT_CLASSES,
                    nexus_head=True, use_checkpointing=False, **kwargs)


def test_nexus_training_step():
    """A training step produces finite nexus losses and gradients for the nexus layers"""
    model = nexus_model()
    model.train()
    loss, metrics = model(nexus_batch(0, 1), stage="train")
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["nexus/edge-loss-train"])
    assert model.core_net.nexus_net.net[0].weight.grad.abs().sum() > 0
    assert model.nexus_decoder.edge_net[0].weight.grad.abs().sum() > 0


def test_nexus_features_and_geometry_training_step():
    """With spacepoint features and geometric messages, gradients reach the new layers"""
    model = nexus_model(nexus_in_features=2, nexus_geometry=True)
    model.train()
    batch = nexus_batch(0, 1, sp_feats=True)
    assert batch["sp"].x.shape == (batch["sp"].num_nodes, 2)
    loss, _ = model(batch, stage="train")
    loss.backward()
    assert torch.isfinite(loss)
    assert model.core_net.nexus_net.edge_net[0].weight.grad.abs().sum() > 0
    assert model.tpc_encoder.nexus_net.weight.grad.abs().sum() > 0


def test_nexus_outputs():
    """Nexus outputs are probabilities, one per spacepoint and one per spacepoint edge"""
    model = nexus_model()
    model.eval()
    batch = nexus_batch(0, 1)
    with torch.no_grad():
        model(batch)
    sp, edge = batch["sp"], batch[E_SP]
    assert sp.x_filter.shape == (sp.num_nodes,)
    assert edge.x.shape == (edge.edge_index.size(1),)
    assert ((sp.x_filter >= 0) & (sp.x_filter <= 1)).all()
    assert ((edge.x >= 0) & (edge.x <= 1)).all()
    events = batch.to_data_list()
    assert events[1][E_SP].x.numel() == events[1][E_SP].edge_index.size(1)


def test_nexus_losses_invariant_to_duplicating_event():
    """A batch holding the same event twice gives the same nexus losses"""
    model = nexus_model()
    model.eval()
    with torch.no_grad():
        _, single = model(nexus_batch(0), stage="val")
        _, double = model(nexus_batch(0, 0), stage="val")
    for key in ("nexus/node-loss-val", "nexus/edge-loss-val"):
        torch.testing.assert_close(double[key], single[key], rtol=1e-4, atol=1e-6)


def test_checkpoint_without_nexus_head_unchanged():
    """Models without the nexus head have no new parameters"""
    model = NuGraph3(in_features=IN_FEATURES, planes=PLANES,
                     semantic_classes=SEMANTIC_CLASSES, event_classes=EVENT_CLASSES)
    assert not any("nexus_net" in k or "nexus_decoder" in k for k in model.state_dict())


def test_vertex_features():
    """Radial edges have |cos| 1 and tangential edges 0; distances are measured from the vertex"""
    from pynuml.data import NuGraphData # pylint: disable=import-outside-toplevel
    from nugraph.models.nugraph3.decoders import NexusDecoder # pylint: disable=import-outside-toplevel
    data = NuGraphData()
    data["sp"].pos = torch.tensor([[10., 0., 0.], [20., 0., 0.], [10., -5., 0.], [10., 5., 0.]])
    data["evt"].y_vtx = torch.zeros(1, 3)
    i, j = torch.tensor([1, 3]), torch.tensor([0, 2]) # radial edge, then tangential edge
    dpos = data["sp"].pos[i] - data["sp"].pos[j]
    r_i, r_j, cos = NexusDecoder(4, vertex="true").vertex_features(data, i, j, dpos)
    torch.testing.assert_close(r_i.squeeze(1), torch.tensor([20., 125. ** 0.5]).log1p())
    torch.testing.assert_close(r_j.squeeze(1), torch.tensor([10., 125. ** 0.5]).log1p())
    torch.testing.assert_close(cos.squeeze(1), torch.tensor([1., 0.]))


def test_nexus_vertex_training_step():
    """Vertex-relative edge features train with the predicted or the true vertex"""
    for vertex, vertex_head in (("pred", True), ("true", False)):
        model = nexus_model(nexus_vertex=vertex, vertex_head=vertex_head)
        model.train()
        loss, _ = model(nexus_batch(0, 1), stage="train")
        loss.backward()
        assert torch.isfinite(loss)
        assert model.nexus_decoder.edge_net[0].in_features == 2 * 32 + 7


def test_nexus_vertex_pred_requires_vertex_head():
    """Using the predicted vertex without the vertex head is an error"""
    try:
        nexus_model(nexus_vertex="pred")
    except RuntimeError:
        return
    raise AssertionError("expected a RuntimeError")


def test_nexus_vote_and_direction_training_step():
    """Vertex votes and spacepoint directions train and feed the edge classifier"""
    for kwargs, extra in (({"nexus_vertex": "vote"}, 3), ({"nexus_direction": True}, 3),
                          ({"nexus_vertex": "vote", "nexus_direction": True}, 6)):
        model = nexus_model(**kwargs)
        model.train()
        batch = nexus_batch(0, 1)
        batch["sp"].y_direction[:5] = torch.tensor([1., 0., 0.]) # synthetic events have few
        loss, metrics = model(batch, stage="train")
        loss.backward()
        assert torch.isfinite(loss)
        assert model.nexus_decoder.edge_net[0].in_features == 2 * 32 + 4 + extra
        if "nexus_vertex" in kwargs:
            assert torch.isfinite(metrics["nexus/vote-resolution-train"])
            assert model.nexus_decoder.vote_net[0].weight.grad.abs().sum() > 0
        if "nexus_direction" in kwargs:
            assert 0 <= metrics["nexus/direction-cos-train"] <= 1
            assert model.nexus_decoder.direction_net[0].weight.grad.abs().sum() > 0


def test_local_shape():
    """Neighbours along a line give a shape concentrated on that axis"""
    from nugraph.models.nugraph3.decoders import NexusDecoder # pylint: disable=import-outside-toplevel
    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.], [2., 0., 0.]])
    i, j = torch.zeros(3, dtype=torch.long), torch.tensor([1, 2, 3])
    shape = NexusDecoder.local_shape(pos, i, j)
    torch.testing.assert_close(shape[0], torch.tensor([1., 0., 0., 0., 0., 0.]))


def test_nexus_charge_training_step():
    """Hit charge and width reach spacepoints and the edge classifier"""
    batch = nexus_batch(0, 1, sp_feats=True, charge=True)
    assert batch["sp"].x.size(1) == 5 and batch["sp"].calo.size(1) == 3
    assert torch.isfinite(batch["sp"].calo).all()
    model = nexus_model(nexus_in_features=5, nexus_charge=True)
    model.train()
    loss, _ = model(batch, stage="train")
    loss.backward()
    assert torch.isfinite(loss)
    assert model.nexus_decoder.edge_net[0].in_features == 2 * 32 + 4 + 3
    assert model.nexus_decoder.edge_net[0].weight.grad[:, -3:].abs().sum() > 0


def test_nexus_knn_shape_and_edge_weight():
    """The direction head can take the stored k-nearest-neighbour shape, and
    up-weighting same-particle edges raises the edge loss"""
    batch = nexus_batch(0, 1)
    assert batch["sp"].local_shape.shape == (batch["sp"].num_nodes, 6)
    model = nexus_model(nexus_direction=True, nexus_knn_shape=True)
    model.train()
    loss, _ = model(batch, stage="train")
    loss.backward()
    assert torch.isfinite(loss)
    assert model.nexus_decoder.direction_net[0].weight.grad.abs().sum() > 0
    edge = []
    for w in (1., 3.):
        model = nexus_model(nexus_edge_pos_weight=w)
        model.eval()
        _, metrics = model(nexus_batch(0, 1), stage="val")
        edge.append(metrics["nexus/edge-loss-val"] - model.nexus_decoder.temp_edge)
    assert edge[1] > edge[0]


def test_nexus_semantic_and_balanced_filter():
    """Semantic predictions feed the edge classifier, and the balanced ghost
    filter loss trains"""
    batch = nexus_batch(0, 1)
    assert batch["sp"].y_semantic.shape == (batch["sp"].num_nodes,)
    model = nexus_model(semantic_head=True, nexus_semantic=True, nexus_balance_filter=True)
    model.train()
    loss, _ = model(batch, stage="train")
    loss.backward()
    assert torch.isfinite(loss)
    assert model.nexus_decoder.edge_net[0].in_features == 2 * 32 + 4 + 2 * len(SEMANTIC_CLASSES)
    assert model.nexus_decoder.node_net.weight.grad.abs().sum() > 0


def test_nexus_log_length_stub_count_and_deeper_edge_classifier():
    """The edge classifier can take log edge lengths and stub counts, and have
    more and wider hidden layers; by default it keeps one hidden layer"""
    batch = nexus_batch(0, 1, graph="gabriel+stubs")
    count = batch["sp"].stub_count
    assert count.shape == (batch["sp"].num_nodes,) and (count >= 0).all()
    assert torch.equal(torch.cat([d["sp"].stub_count for d in batch.to_data_list()]), count)
    model = nexus_model(nexus_log_length=True, nexus_stub_count=True,
                        nexus_edge_layers=2, nexus_edge_width=48)
    model.train()
    loss, _ = model(batch, stage="train")
    loss.backward()
    assert torch.isfinite(loss)
    net = model.nexus_decoder.edge_net
    assert net[0].in_features == 2 * 32 + 4 + 3 and net[0].out_features == 48 and len(net) == 5
    assert net[0].weight.grad.abs().sum() > 0
    assert len(nexus_model().nexus_decoder.edge_net) == 3
