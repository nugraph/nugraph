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


def test_spacepoint_direction():
    """Spacepoints along a straight particle get that particle's direction"""
    data = NuGraphData()
    n = 6
    line = torch.arange(n, dtype=torch.float)[:, None] * torch.tensor([[1., 2., 2.]]) / 3.
    data["hit"].y_semantic = torch.zeros(n, dtype=torch.long)
    data["particle-truth"].num_nodes = 1
    data["hit", "cluster-truth", "particle-truth"].edge_index = torch.stack(
        (torch.arange(n), torch.zeros(n, dtype=torch.long)))
    data["sp"].pos = line
    data["hit", "nexus", "sp"].edge_index = torch.stack((torch.arange(n), torch.arange(n)))
    data = SpacePointGraph(k=3)(data)
    cos = (data["sp"].y_direction @ torch.tensor([1., 2., 2.]) / 3.).abs()
    torch.testing.assert_close(cos, torch.ones(n))


def test_majority_labels():
    """Three-hit spacepoints with two hits from one particle and an odd hit
    carrying truth take that particle; noise, two-hit and split cases do not"""
    data = NuGraphData()
    # hits: 0-1 particle 0, 2 particle 1, 3 diffuse (no instance), 4 noise, 5 particle 2
    data["hit"].y_semantic = torch.tensor([0, 0, 1, 6, -1, 2])
    data["particle-truth"].num_nodes = 3
    data["hit", "cluster-truth", "particle-truth"].edge_index = torch.tensor([[0, 1, 2, 5], [0, 0, 1, 2]])
    data["sp"].pos = torch.arange(15, dtype=torch.float).reshape(5, 3)
    data["hit", "nexus", "sp"].edge_index = torch.tensor(
        [[0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 2, 0, 2, 5],
         [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4]])
    plain = SpacePointGraph(k=2, majority=False)(data.clone())["sp"].y_instance
    majority = SpacePointGraph(k=2)(data)["sp"].y_instance
    assert plain.tolist() == [-1, -1, -1, -1, -1]
    assert majority.tolist() == [0, 0, -1, -1, -1]


def test_gabriel_edges():
    """With every spacepoint a candidate, Gabriel edges are exactly the pairs
    whose diametral sphere holds no other spacepoint"""
    torch.manual_seed(0)
    pos = torch.rand(40, 3) * 10.
    data = NuGraphData()
    data["sp"].pos = pos
    data = SpacePointGraph(k=4, graph="gabriel", candidates=39)(data)
    found = set(map(tuple, data[E_SP].edge_index.T.tolist()))
    expected = set()
    for i in range(40):
        for j in range(40):
            if i == j:
                continue
            mid, r = 0.5 * (pos[i] + pos[j]), 0.5 * (pos[i] - pos[j]).norm()
            d = (pos - mid).norm(dim=1)
            d[[i, j]] = torch.inf
            if (d >= r).all():
                expected.add((i, j))
    assert found == expected


def test_stub_edges_follow_lines():
    """Spacepoints on two parallel lines, farther apart than the stub reach,
    get stub edges to their two nearest neighbours each way along their own
    line, where enough points lie that way to pass the vote threshold"""
    t = torch.arange(20, dtype=torch.float)[:, None] * torch.tensor([[1., 0., 0.]])
    pos = torch.cat((t, t + torch.tensor([[0., 30., 0.]])))
    data = NuGraphData()
    data["sp"].pos = pos
    transform = SpacePointGraph(k=4, graph="gabriel+stubs")
    nbr, dist = transform.neighbours(pos, torch.zeros(40, dtype=torch.long), 64)
    found = set(map(tuple, transform.stubs(pos, nbr, dist).T.tolist()))
    expected = {(i, j) for i in range(40) for j in range(40)
                if i // 20 == j // 20 and abs(i - j) in (1, 2)
                and (i % 20 >= 4 if j < i else i % 20 <= 15)}
    assert found == expected


def test_batched_graph():
    """A batch of events gets the same graphs and labels as each event alone"""
    from torch_geometric.data import Batch
    for graph in ("knn", "gabriel+stubs"):
        transform = SpacePointGraph(k=4, graph=graph, chunk_size=7)
        events = [make_event(seed) for seed in range(3)]
        batch = transform(Batch.from_data_list([e.clone() for e in events]))
        for e, split in zip(events, batch.to_data_list()):
            e = transform(e.clone())
            assert set(map(tuple, split[E_SP].edge_index.T.tolist())) == \
                set(map(tuple, e[E_SP].edge_index.T.tolist()))
            assert torch.equal(split["sp"].y_instance, e["sp"].y_instance)
        offset = 0
        for e in events:
            e = transform(e)
            n = e["sp"].num_nodes
            src, dst = batch[E_SP].edge_index
            mine = (dst >= offset) & (dst < offset + n)
            assert not ((src[mine] < offset) | (src[mine] >= offset + n)).any()
            got = set(map(tuple, (batch[E_SP].edge_index[:, mine] - offset).T.tolist()))
            assert got == set(map(tuple, e[E_SP].edge_index.T.tolist()))
            torch.testing.assert_close(batch["sp"].y_direction[offset:offset + n], e["sp"].y_direction)
            offset += n


def ancestry_event() -> NuGraphData:
    """
    A muon (instance 0) with three delta rays: a short one (1), a long one (2)
    and a short delta ray of that long one (3), each owning one hit and one
    spacepoint
    """
    data = NuGraphData()
    data["hit"].num_nodes = 4
    data["hit"].y_semantic = torch.zeros(4, dtype=torch.long)
    pt = data["particle-truth"]
    pt.num_nodes = 4
    pt.g4_id = torch.tensor([1, 2, 3, 4])
    pt.parent_g4_id = torch.tensor([0, 1, 1, 3])
    pt.pdg_code = torch.tensor([13, 11, -11, 11])
    pt.start_process = torch.tensor([0, 2, 2, 4])
    pt.start_position = torch.zeros(4, 3)
    pt.end_position = torch.tensor([[100., 0., 0.], [1., 0., 0.], [10., 0., 0.], [0., 0.5, 0.]])
    data["hit", "cluster-truth", "particle-truth"].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    data["sp"].pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    data["hit", "nexus", "sp"].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    return data


def test_merge_short_delta_rays():
    """Delta rays shorter than the threshold take their parent's label,
    through chains of delta rays; longer ones keep their own"""
    assert SpacePointGraph(k=2)(ancestry_event())["sp"].y_instance.tolist() == [0, 1, 2, 3]
    merged = SpacePointGraph(k=2, merge_delta=2.)(ancestry_event())
    assert merged["sp"].y_instance.tolist() == [0, 0, 2, 2]
    assert SpacePointGraph(k=2, merge_delta=20.)(ancestry_event())["sp"].y_instance.tolist() == [0, 0, 0, 0]
    # within a batch, each event's delta rays merge into their own event's parents
    from torch_geometric.data import Batch
    batch = SpacePointGraph(k=2, merge_delta=2.)(Batch.from_data_list([ancestry_event(), ancestry_event()]))
    assert batch["sp"].y_instance.tolist() == [0, 0, 2, 2] * 2


def test_local_shape_from_nearest_neighbours():
    """Spacepoints on a line have a k-nearest-neighbour shape along that line,
    whatever graph is built"""
    pos = torch.arange(10, dtype=torch.float)[:, None] * torch.tensor([[0., 0., 1.]])
    data = NuGraphData()
    data["sp"].pos = pos
    for graph in ("knn", "gabriel+stubs"):
        shape = SpacePointGraph(k=4, graph=graph)(data.clone())["sp"].local_shape
        torch.testing.assert_close(shape, torch.tensor([[0., 0., 0., 0., 0., 1.]]).expand(10, 6))


def test_spacepoint_semantic_label():
    """Each spacepoint takes the majority semantic class of its labelled hits"""
    data = SpacePointGraph(k=3)(toy_event())
    # hit classes [0, 0, 1, -1]; spacepoints hold hits {0,1}, {0,2}, {3}, {1}
    assert data["sp"].y_semantic.tolist()[0] == 0
    assert data["sp"].y_semantic.tolist()[2] == -1
    assert data["sp"].y_semantic.tolist()[3] == 0


def test_extra_knn_edges():
    """Extra nearest-neighbour edges join the Gabriel graph in both directions"""
    data = make_event(0)
    plain = SpacePointGraph(k=4, graph="gabriel")(data.clone())
    extra = SpacePointGraph(k=4, graph="gabriel", extra_knn=3)(data.clone())
    pos = data["sp"].pos
    found = set(map(tuple, extra[E_SP].edge_index.T.tolist()))
    assert set(map(tuple, plain[E_SP].edge_index.T.tolist())) <= found
    nearest = torch.cdist(pos, pos).fill_diagonal_(torch.inf).topk(3, largest=False).indices
    for i in range(pos.size(0)):
        for j in nearest[i].tolist():
            assert (i, j) in found and (j, i) in found
