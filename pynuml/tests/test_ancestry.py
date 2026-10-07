"""Tests for particle ancestry in HitGraphProducer"""
import numpy as np
import pandas as pd
import torch

from pynuml.process import HitGraphProducer
from pynuml.process.hitgraph import PROCESSES

from test_regression_pynuml import FakeEvent, FakeFile

E_PARENT = ("particle-truth", "parent", "particle-truth")

# g4_id, parent_id, pdg, start process, end process, has hits
PARTICLES = [
    (1, 0, 13, "primary", "CoupledTransportation", True),  # muon
    (2, 0, 211, "primary", "Decay", True),                  # pion ...
    (3, 2, -13, "Decay", "Decay", True),                    # ... decays to a muon ...
    (4, 3, -11, "Decay", "eIoni", True),                    # ... which decays to a Michel electron
    (5, 0, 111, "primary", "Decay", False),                 # neutral pion, no hits
    (6, 5, 22, "Decay", "conv", True),                      # its photon
    (8, 1, 2112, "hadElastic", "CoupledTransportation", False),  # hitless neutron from the muon
    (9, 8, 2212, "hadElastic", "hIoni", True),              # proton knocked out by that neutron
]


def labeller(particles: pd.DataFrame) -> pd.DataFrame:
    """Label every particle with semantic class 0 and its own instance"""
    out = particles[["g4_id", "parent_id", "type", "momentum", "start_process", "end_process"]].copy()
    out["semantic_label"] = 0
    out["instance_g4_id"] = out["g4_id"]
    return out
labeller.labels = ["track", "invisible"]


def ancestry_event(seed: int = 0) -> FakeEvent:
    """Hits ordered by hit_id, two hits per plane for every particle with hits"""
    rng = np.random.default_rng(seed)
    with_hits = [p[0] for p in PARTICLES if p[5]]
    rows, edeps = [], []
    for plane in range(3):
        for g in with_hits:
            for _ in range(2):
                h = len(rows)
                rows.append((h, rng.uniform(10, 50), rng.uniform(1, 3), 0, plane,
                             rng.uniform(0, 100), rng.uniform(0, 100)))
                edeps.append((h, g, rng.uniform(0.1, 1.)))
    hits = pd.DataFrame(rows, columns=["hit_id", "integral", "rms", "tpc", "view", "proj", "drift"])
    edeps = pd.DataFrame(edeps, columns=["hit_id", "g4_id", "energy"])
    by_plane = [hits.hit_id[hits.view == p].to_numpy() for p in range(3)]
    spacepoints = pd.DataFrame({
        "position_x": rng.uniform(0, 100, 6), "position_y": rng.uniform(0, 100, 6),
        "position_z": rng.uniform(0, 100, 6),
        "hit_id_u": by_plane[0][:6], "hit_id_v": by_plane[1][:6], "hit_id_y": by_plane[2][:6]})
    particles = pd.DataFrame(
        [(g, p, t, 1., s, e, g, 2. * g, 3. * g, g + .5, 2. * g + .5, 3. * g + .5)
         for g, p, t, s, e, _ in PARTICLES],
        columns=["g4_id", "parent_id", "type", "momentum", "start_process", "end_process",
                 "start_position_x", "start_position_y", "start_position_z",
                 "end_position_x", "end_position_y", "end_position_z"])
    return FakeEvent({"hit_table": hits, "spacepoint_table": spacepoints,
                      "edep_table": edeps, "particle_table": particles})


def produce(**kwargs):
    producer = HitGraphProducer(FakeFile(), semantic_labeller=labeller, lower_bound=1,
                                ancestry=True, **kwargs)
    _, data = producer(ancestry_event())
    return data


def test_parent_edges():
    """Particles link to their nearest stored ancestor, skipping hitless ones"""
    data = produce()
    g4 = data["particle-truth"].g4_id.tolist()
    assert sorted(g4) == [1, 2, 3, 4, 6, 9]
    child, parent = data[E_PARENT].edge_index
    links = {(g4[c], g4[p]): int(n) for c, p, n in zip(child, parent, data[E_PARENT].generations)}
    assert links == {(3, 2): 1, (4, 3): 1, (9, 1): 2}


def test_particle_truth():
    """Raw parent, processes and start/end positions are stored per particle"""
    pt = produce()["particle-truth"]
    i = pt.g4_id.tolist().index(6)
    assert int(pt.parent_g4_id[i]) == 5
    assert PROCESSES[int(pt.start_process[i])] == "Decay"
    assert PROCESSES[int(pt.end_process[i])] == "conv"
    torch.testing.assert_close(pt.start_position[i], torch.tensor([6., 12., 18.]))
    torch.testing.assert_close(pt.end_position[i], torch.tensor([6.5, 12.5, 18.5]))


def test_ancestry_off_by_default():
    """Without ancestry the graph has no parent edges or extra particle truth"""
    producer = HitGraphProducer(FakeFile(), semantic_labeller=labeller, lower_bound=1)
    _, data = producer(ancestry_event())
    assert E_PARENT not in data.edge_types
    assert "start_position" not in data["particle-truth"]


def test_save_and_load(tmp_path):
    """Ancestry survives the graph file format"""
    import h5py # pylint: disable=import-outside-toplevel
    from pynuml.data import NuGraphData # pylint: disable=import-outside-toplevel
    data = produce()
    with h5py.File(tmp_path / "graph.h5", "w") as f:
        data.save(f, "event")
    with h5py.File(tmp_path / "graph.h5") as f:
        loaded = NuGraphData.load(f["event"])
    assert torch.equal(loaded[E_PARENT].edge_index, data[E_PARENT].edge_index)
    assert torch.equal(loaded[E_PARENT].generations, data[E_PARENT].generations)
    torch.testing.assert_close(loaded["particle-truth"].start_position,
                               data["particle-truth"].start_position)


def test_split_delta_rays():
    """Delta rays join their parent's instance by default and get their own when split"""
    from pynuml.labels import StandardLabels # pylint: disable=import-outside-toplevel
    particles = pd.DataFrame({
        "g4_id": [1, 2], "parent_id": [0, 1], "type": [13, 11], "momentum": [1.0, 0.05],
        "start_process": ["primary", "muIoni"], "end_process": ["CoupledTransportation", "eIoni"]})
    merged = StandardLabels()(particles).set_index("g4_id")
    split = StandardLabels(split_delta_rays=True)(particles).set_index("g4_id")
    assert merged.instance_g4_id[2] == 1
    assert split.instance_g4_id[2] == 2
    assert merged.semantic_label[2] == split.semantic_label[2]


def test_load_without_parent_edges(tmp_path):
    """Graphs without parent edges load with an empty generations tensor and batch"""
    import h5py # pylint: disable=import-outside-toplevel
    from torch_geometric.data import Batch # pylint: disable=import-outside-toplevel
    from pynuml.data import NuGraphData # pylint: disable=import-outside-toplevel
    data = produce()
    empty = data.clone()
    empty[E_PARENT].edge_index = torch.empty((2, 0), dtype=torch.long)
    empty[E_PARENT].generations = torch.empty(0, dtype=torch.long)
    with h5py.File(tmp_path / "graph.h5", "w") as f:
        data.save(f, "full")
        empty.save(f, "empty")
    with h5py.File(tmp_path / "graph.h5") as f:
        full, loaded = NuGraphData.load(f["full"]), NuGraphData.load(f["empty"])
    assert loaded[E_PARENT].generations.shape == (0,)
    batch = Batch.from_data_list([full, loaded])
    assert batch[E_PARENT].generations.shape == (3,)
