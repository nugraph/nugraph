"""Phase 0 regression tests for pynuml processing and labels

Tests for known, unfixed bugs are marked ``xfail(strict=True)`` with the bug
number from the roadmap's Phase 0 tracker; whoever fixes a bug removes its
marker in the same PR.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from pynuml.labels import FlavorLabels, PDKLabels, StandardLabels
from pynuml.process import HitGraphProducer


def bug(number: int, text: str):
    """Mark a test as an expected failure caused by a tracked bug"""
    return pytest.mark.xfail(strict=True, reason=f"bug {number}: {text}")


@bug(7, "label() range check references an undefined name `label` instead of `idx`")
@pytest.mark.parametrize("labeller, name", [
    (StandardLabels(), "pion"),
    (FlavorLabels(), "cc_nue"),
    (PDKLabels(), "nu"),
])
def test_label_lookup(labeller, name):
    assert labeller.label(0) == name


def test_label_index_round_trip():
    labeller = StandardLabels()
    for i, name in enumerate(labeller.labels):
        assert labeller.index(name) == i


class FakeFile: # pylint: disable=too-few-public-methods
    """Stands in for pynuml.io.File; the producer only registers columns"""
    def add_group(self, *_):
        pass


class FakeEvent:
    """Stands in for pynuml.io.Event: a name, an ID and a dict of tables"""
    def __init__(self, tables: dict[str, pd.DataFrame]):
        self.name = "r0_sr0_evt0"
        self.event_id = (0, 0, 0)
        self._tables = tables

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self._tables[key].copy()


def fake_labeller(particles: pd.DataFrame) -> pd.DataFrame:
    """Label every particle as its own instance with semantic class 0"""
    out = particles[["g4_id", "type", "momentum"]].copy()
    out["semantic_label"] = 0
    out["instance_g4_id"] = out["g4_id"]
    return out
fake_labeller.labels = ["track", "invisible"]


def shuffled_event(seed: int = 0) -> tuple[FakeEvent, dict[int, int]]:
    """
    An event whose hit table is NOT ordered by hit_id

    Returns the event and the true g4_id of every hit_id.
    """
    rng = np.random.default_rng(seed)
    n_per_plane, planes = 8, 3
    n_hit = n_per_plane * planes
    hit_ids = rng.permutation(n_hit) # row order != hit_id
    hits = pd.DataFrame({
        "hit_id": hit_ids,
        "integral": rng.uniform(10, 50, n_hit),
        "rms": rng.uniform(1, 3, n_hit),
        "tpc": np.zeros(n_hit, dtype=int),
        "view": np.repeat(np.arange(planes), n_per_plane),
        "proj": rng.uniform(0, 100, n_hit),
        "drift": rng.uniform(0, 100, n_hit),
    })
    truth = {int(h): 1 + int(h) % 2 for h in hit_ids} # g4_id 1 or 2 by parity
    edeps = pd.DataFrame({
        "hit_id": hit_ids,
        "g4_id": [truth[int(h)] for h in hit_ids],
        "energy": rng.uniform(0.1, 1.0, n_hit),
    })
    particles = pd.DataFrame({
        "g4_id": [1, 2], "parent_id": [0, 0], "type": [13, 211],
        "momentum": [1.0, 0.5], "start_process": ["primary"] * 2,
        "end_process": ["CoupledTransportation"] * 2,
    })
    by_plane = [hits.hit_id[hits.view == p].to_numpy() for p in range(planes)]
    spacepoints = pd.DataFrame({
        "position_x": rng.uniform(0, 100, 5),
        "position_y": rng.uniform(0, 100, 5),
        "position_z": rng.uniform(0, 100, 5),
        "hit_id_u": by_plane[0][:5],
        "hit_id_v": by_plane[1][:5],
        "hit_id_y": by_plane[2][:5],
    })
    tables = {"hit_table": hits, "spacepoint_table": spacepoints,
              "edep_table": edeps, "particle_table": particles}
    return FakeEvent(tables), truth


@bug(6, "truth hit->particle edges use hit_id as the node index, "
        "so labels are wrong whenever hit rows are not ordered by hit_id")
def test_truth_edges_follow_hit_rows():
    """Each hit node's true particle must match its own hit_id's energy deposit"""
    evt, truth = shuffled_event()
    producer = HitGraphProducer(FakeFile(), semantic_labeller=fake_labeller,
                                lower_bound=1)
    _, data = producer(evt)
    assert data is not None
    y_i = data.y_i()
    g4 = data["particle-truth"].g4_id
    for row, hit_id in enumerate(data["hit"].id.tolist()):
        assert int(g4[y_i[row]]) == truth[hit_id], f"hit row {row} (hit_id {hit_id})"


def test_truth_edges_with_ordered_hits():
    """Control: with hit rows ordered by hit_id the truth edges are correct today"""
    evt, truth = shuffled_event()
    for key in ("hit_table", "edep_table"):
        evt._tables[key] = evt._tables[key].sort_values("hit_id").reset_index(drop=True) # pylint: disable=protected-access
    producer = HitGraphProducer(FakeFile(), semantic_labeller=fake_labeller,
                                lower_bound=1)
    _, data = producer(evt)
    y_i = data.y_i()
    g4 = data["particle-truth"].g4_id
    for row, hit_id in enumerate(data["hit"].id.tolist()):
        assert int(g4[y_i[row]]) == truth[hit_id]
    assert torch.equal(data["hit"].id, torch.arange(data["hit"].num_nodes))
