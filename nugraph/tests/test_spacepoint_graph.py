"""Tests for the SpacePointGraph transform"""
import torch

from nugraph.util import SpacePointGraph
from pynuml.data import NuGraphData

from synthetic import make_event

E_SP = ("sp", "sp3d", "sp")


def test_knn_edges():
    """Each spacepoint receives edges from exactly its k nearest neighbours"""
    data = SpacePointGraph(k=4, chunk_size=7)(make_event(0))
    pos = data["sp"].pos
    src, dst = data[E_SP].edge_index
    n = pos.size(0)
    assert torch.equal(dst.bincount(minlength=n), torch.full((n,), 4))
    assert not (src == dst).any()
    dist = torch.cdist(pos, pos).fill_diagonal_(torch.inf)
    expected = dist.topk(4, largest=False).indices
    for i in range(n):
        assert set(src[dst == i].tolist()) == set(expected[i].tolist())


def toy_event() -> NuGraphData:
    """
    Four hits and four spacepoints with known truth

    Hits 0 and 1 belong to particle 0, hit 2 to particle 1 and hit 3 is
    background. Spacepoint 0 uses hits 0 and 1 (pure), spacepoint 1 hits 0 and
    2 (mixed), spacepoint 2 hit 3 (background) and spacepoint 3 hit 1 (pure).
    """
    data = NuGraphData()
    data["hit"].y_semantic = torch.tensor([0, 0, 1, -1])
    data["particle-truth"].num_nodes = 2
    data["hit", "cluster-truth", "particle-truth"].edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
    data["sp"].pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [5., 0., 0.], [0., 1., 0.]])
    data["hit", "nexus", "sp"].edge_index = torch.tensor([[0, 1, 0, 2, 3, 1], [0, 0, 1, 1, 2, 3]])
    return data


def test_spacepoint_truth():
    """Spacepoints are labelled only when all of their hits share one particle"""
    data = SpacePointGraph(k=3)(toy_event())
    assert data["sp"].y_instance.tolist() == [0, -1, -1, 0]


def test_edge_truth():
    """Edges are labelled 1 only between spacepoints of the same particle"""
    data = SpacePointGraph(k=3)(toy_event())
    src, dst = data[E_SP].edge_index
    pairs = {(int(s), int(d)): int(y) for s, d, y in zip(src, dst, data[E_SP].y)}
    assert pairs[(0, 3)] == 1 and pairs[(3, 0)] == 1
    assert sum(pairs.values()) == 2


def test_no_truth():
    """Without true instances the graph is built and no labels are added"""
    data = toy_event()
    del data["hit", "cluster-truth", "particle-truth"]
    data = SpacePointGraph(k=2)(data)
    assert data[E_SP].edge_index.size(1) == 8
    assert "y_instance" not in data["sp"]
    assert "y" not in data[E_SP]


def test_single_spacepoint():
    """An event with one spacepoint gets an empty spacepoint graph"""
    data = NuGraphData()
    data["sp"].pos = torch.zeros(1, 3)
    data = SpacePointGraph(k=4)(data)
    assert data[E_SP].edge_index.shape == (2, 0)
