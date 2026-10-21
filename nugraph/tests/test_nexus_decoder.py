"""Tests for the NuGraph3 nexus decoder"""
import torch
from torch_geometric.data import Batch

from nugraph.models.nugraph3 import NuGraph3
from nugraph.models.nugraph3.transform import Transform
from nugraph.util import NexusFeatures, SpacePointGraph

from synthetic import EVENT_CLASSES, IN_FEATURES, PLANES, SEMANTIC_CLASSES, make_event

E_SP = ("sp", "sp3d", "sp")


def nexus_batch(*seeds: int, sp_feats: bool = False) -> Batch:
    """Batch of synthetic events with spacepoint graph and truth labels"""
    transforms = [Transform(PLANES), SpacePointGraph(k=4)]
    if sp_feats:
        transforms.append(NexusFeatures(PLANES, positions=False))
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
