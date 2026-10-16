"""Tests for spacepoint candidates built from hit crossings"""
import torch

from pynuml.data import NuGraphData
from nugraph.util.spacepoint_builder import SpacePointBuilder

# MicroBooNE planes u, v, y: wire axis (direction, origin, resolution), then drift axis
GEOMETRY = [[0., -0.8660254, 0.5, 101.44, 0.3, 1., 0., 0., 44., 0.055],
            [0., 0.8660254, 0.5, 99.76, 0.3, 1., 0., 0., 44., 0.055],
            [0., 0., 1., -0.25, 0.3, 1., 0., 0., 44., 0.055]]
VOLUME = [[0., -116.5, 0.], [256.35, 116.5, 1036.8]]


def event(points: list, skip: tuple = ()) -> NuGraphData:
    """Hits of 3D points projected onto the three planes; skip lists (point, plane) pairs with no hit"""
    g = torch.tensor(GEOMETRY)
    plane, pos = [], []
    for i, r in enumerate(torch.tensor(points)):
        for p in range(3):
            if (i, p) not in skip:
                plane.append(p)
                pos.append([float(r @ g[p, 0:3] + g[p, 3]), float(r @ g[p, 5:8] + g[p, 8])])
    data = NuGraphData()
    data["hit"].plane = torch.tensor(plane)
    data["hit"].pos = torch.tensor(pos)
    data["sp"].pos = torch.zeros(5, 3)
    data["hit", "nexus", "sp"].edge_index = torch.zeros(2, 0, dtype=torch.long)
    return data


def candidates(data):
    sp = data["sp"].pos
    hit, s = data["hit", "nexus", "sp"].edge_index
    return sp, [hit[s == i].tolist() for i in range(len(sp))]


def test_one_point_gives_one_triplet():
    """A point seen on all three planes gives one triplet at that point, linked to its three hits"""
    data = SpacePointBuilder(GEOMETRY, VOLUME)(event([[100., 20., 500.]]))
    sp, hits = candidates(data)
    assert len(sp) == 1 and hits == [[0, 1, 2]]
    assert torch.allclose(sp[0], torch.tensor([100., 20., 500.]), atol=1e-3)


def test_missing_third_hit_gives_doublet():
    """A point with no hit on the third plane gives a two-plane crossing at the point"""
    data = SpacePointBuilder(GEOMETRY, VOLUME)(event([[100., 20., 500.]], skip={(0, 2)}))
    sp, hits = candidates(data)
    assert len(sp) == 1 and hits == [[0, 1]]
    assert torch.allclose(sp[0], torch.tensor([100., 20., 500.]), atol=1e-3)


def test_points_at_one_drift_time_make_ghosts():
    """Two points at the same drift coordinate cross in ghosts too; both true points are among the
    triplets, and doublets are only made from hits no triplet uses"""
    pts = [[100., 20., 500.], [100., -60., 300.]]
    data = SpacePointBuilder(GEOMETRY, VOLUME)(event(pts))
    sp, hits = candidates(data)
    for r in torch.tensor(pts):
        assert (sp - r).norm(dim=1).min() < 1e-3
    assert all(len(h) == 3 for h in hits)


def test_far_drift_or_outside_volume_gives_nothing():
    """Hits whose drift coordinates differ by more than the window, or that cross outside the
    volume, give no candidate"""
    data = event([[100., 20., 500.], [150., 20., 600.]], skip={(0, 1), (0, 2), (1, 0), (1, 2)})
    sp, _ = candidates(SpacePointBuilder(GEOMETRY, VOLUME)(data))
    assert len(sp) == 0
    data = event([[100., 20., 500.]], skip={(0, 2)})
    sp, _ = candidates(SpacePointBuilder(GEOMETRY, [[0., -116.5, 0.], [256.35, 116.5, 400.]])(data))
    assert len(sp) == 0


def test_cap_keeps_triplets_first():
    """With more candidates than the cap, triplets are kept before doublets"""
    pts = [[100., 20., 500.], [100., -60., 300.], [100., 0., 700.]]
    data = event(pts, skip={(2, 2)})
    full, _ = candidates(SpacePointBuilder(GEOMETRY, VOLUME)(event(pts, skip={(2, 2)})))
    sp, hits = candidates(SpacePointBuilder(GEOMETRY, VOLUME, cap=2)(data))
    assert len(full) > 2 and len(sp) == 2 and all(len(h) == 3 for h in hits)


def test_event_links_follow_candidates():
    """Spacepoint-to-event links made for the file's spacepoints are remade for the candidates"""
    data = event([[100., 20., 500.]])
    data["sp", "in", "evt"].edge_index = torch.stack((torch.arange(5), torch.zeros(5, dtype=torch.long)))
    data = SpacePointBuilder(GEOMETRY, VOLUME)(data)
    assert data["sp", "in", "evt"].edge_index.tolist() == [[0], [0]]


def test_best_keeps_each_hits_closest_partner():
    """Two points 0.15 cm apart in drift cross in two ghosts as well; keeping each hit's best
    candidate drops the ghosts, whose hits all have a closer partner"""
    data = event([[100., 20., 500.], [100.15, -20., 520.]], skip={(0, 2), (1, 2)})
    full, _ = candidates(SpacePointBuilder(GEOMETRY, VOLUME)(data.clone()))
    sp, hits = candidates(SpacePointBuilder(GEOMETRY, VOLUME, best=1)(data))
    assert len(full) == 4
    assert sorted(hits) == [[0, 1], [2, 3]]
