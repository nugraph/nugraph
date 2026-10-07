"""Tests for the beta-seeded instance materializer"""
import argparse
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch

from nugraph.models.nugraph3 import NuGraph3
from nugraph.models.nugraph3.decoders import InstanceDecoder
from nugraph.models.nugraph3.decoders import instance as instance_module
from pynuml.data import NuGraphData

from synthetic import EVENT_CLASSES, IN_FEATURES, PLANES, SEMANTIC_CLASSES, make_batch


def decoder(**kwargs) -> InstanceDecoder:
    """Instance decoder using the seed materializer"""
    return InstanceDecoder(beta_features=32, coord_features=128, instance_features=8,
                           semantic_classes=SEMANTIC_CLASSES, materializer="seed", **kwargs)


def seeded_hits(seed: int = 0) -> NuGraphData:
    """One graph with three tight clusters, each with a single high-beta hit"""
    g = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[0.] * 8, [5.] * 8, [-5.] * 8])
    data = NuGraphData()
    h = data["hit"]
    h.x = torch.zeros(30, 4)
    h.ox = centres.repeat_interleave(10, dim=0) + 0.01 * torch.randn(30, 8, generator=g)
    h.of = torch.full((30,), 0.1)
    h.of[[0, 10, 20]] = 0.9
    h.y_semantic = torch.zeros(30, dtype=torch.long)
    return data


def test_seed_finds_condensation_points():
    """Each cluster with a high-beta hit becomes one particle holding all its hits"""
    data = seeded_hits()
    decoder().materialize(data)
    assert data["particle"].x.size(0) == 3
    torch.testing.assert_close(data.x_i(), torch.arange(3).repeat_interleave(10))


def test_seed_suppresses_nearby_candidates():
    """A lower-beta candidate within the seed radius of another does not start a particle"""
    data = seeded_hits()
    data["hit"].of[1] = 0.8
    decoder().materialize(data)
    assert data["particle"].x.size(0) == 3
    decoder(seed_radius=0.).materialize(data)
    assert data["particle"].x.size(0) == 4


def test_seed_without_candidates():
    """Without any hit above the beta threshold, no particles are formed"""
    data = seeded_hits()
    decoder(seed_beta=0.95).materialize(data)
    assert data["particle"].x.size(0) == 0
    assert data["hit", "cluster", "particle"].edge_index.size(1) == 0


def test_seed_assign_radius():
    """Hits beyond the assignment radius stay unassigned; by default every hit is assigned"""
    data = seeded_hits()
    data["hit"].ox[5] += 2.
    decoder().materialize(data)
    assert (data.x_i() > -1).all()
    decoder(seed_assign_radius=1.).materialize(data)
    assert data.x_i()[5] == -1 and (data.x_i() > -1).sum() == 29


def test_seed_batch_matches_single_graphs():
    """Batched materialization gives each graph the same particles as on its own"""
    graphs = [seeded_hits(0), seeded_hits(1)]
    graphs[1]["hit"].of[10] = 0.1
    batch = Batch.from_data_list(graphs)
    decoder().materialize(batch)
    torch.testing.assert_close(batch["particle"].batch, torch.tensor([0, 0, 0, 1, 1]))
    for single, batched in zip(graphs, batch.to_data_list()):
        decoder().materialize(single)
        torch.testing.assert_close(batched.x_i(), single.x_i())


def test_seed_independent_of_block_size(monkeypatch):
    """Computing distances in small blocks gives the same particles"""
    data = seeded_hits()
    data["hit"].of[[1, 11, 12]] = 0.8
    decoder().materialize(data)
    expected = data.x_i()
    monkeypatch.setattr(instance_module, "MAX_PAIRS", 5)
    decoder().materialize(data)
    torch.testing.assert_close(data.x_i(), expected)


def test_materializer_options():
    """DBSCAN stays the default, unknown methods are rejected, and CLI options reach the decoder"""
    assert InstanceDecoder(32, 128, 8, SEMANTIC_CLASSES).materializer == "dbscan"
    with pytest.raises(ValueError):
        InstanceDecoder(32, 128, 8, SEMANTIC_CLASSES, materializer="graph")
    parser = NuGraph3.add_model_args(argparse.ArgumentParser())
    args = parser.parse_args(["--instance", "--materializer", "seed", "--seed-beta", "0.7"])
    nudata = SimpleNamespace(planes=PLANES, semantic_classes=SEMANTIC_CLASSES,
                             event_classes=EVENT_CLASSES)
    model = NuGraph3.from_args(args, nudata)
    dec = model.instance_decoder
    assert (dec.materializer, dec.seed_beta, dec.seed_radius) == ("seed", 0.7, 0.3)
    assert dec.seed_assign_radius == float("inf")


def test_seed_validation_step():
    """A full evaluation pass with the seed materializer produces a valid ARI"""
    torch.manual_seed(0)
    model = NuGraph3(in_features=IN_FEATURES, planes=PLANES, semantic_classes=SEMANTIC_CLASSES,
                     event_classes=EVENT_CLASSES, instance_head=True, use_checkpointing=False,
                     materializer="seed", seed_beta=0.)
    model.train()
    with torch.no_grad():
        model(make_batch(0, 1)) # prime the running input norm
    model.eval()
    batch = make_batch(0, 1)
    with torch.no_grad():
        _, metrics = model(batch, stage="val")
    assert -1. <= metrics["instance/adjusted-rand-val"] <= 1.
    assert batch["particle"].batch.size(0) == batch["particle"].x.size(0)
