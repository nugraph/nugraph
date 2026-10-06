"""Phase 0 regression tests for NuGraph3

Each test states the intended behaviour. Tests for known, unfixed bugs are
marked ``xfail(strict=True)`` with the bug number from the roadmap's Phase 0
tracker, so the suite passes today and turns red (XPASS) when a fix lands;
whoever fixes the bug removes the marker in the same PR.
"""
import importlib.util
import pathlib

import pytest
import torch
from torch_geometric.data import Batch

from nugraph.models.nugraph3 import NuGraph3
from nugraph.models.nugraph3.decoders import InstanceDecoder
from nugraph.data.BalanceSampler import BalanceSampler
from nugraph.util import FeatureExtension, InputNorm, ObjConLoss
from pynuml.data import NuGraphData

from synthetic import (EVENT_CLASSES, IN_FEATURES, PLANES, SEMANTIC_CLASSES,
                       make_batch)

REPO = pathlib.Path(__file__).resolve().parents[2]


def bug(tag: int | str, text: str):
    """Mark a test as an expected failure caused by a tracked bug"""
    tag = f"bug {tag}" if isinstance(tag, int) else tag
    return pytest.mark.xfail(strict=True, reason=f"{tag}: {text}")


def build_model(**kwargs) -> NuGraph3:
    """NuGraph3 with every TPC head on, input norm primed on synthetic data"""
    torch.manual_seed(0)
    args = dict(in_features=IN_FEATURES, planes=PLANES,
                semantic_classes=SEMANTIC_CLASSES, event_classes=EVENT_CLASSES,
                event_head=True, semantic_head=True, filter_head=True,
                vertex_head=True, instance_head=True, spacepoint_head=True,
                use_checkpointing=False)
    args.update(kwargs)
    model = NuGraph3(**args)

    # prime the running input norm so evaluation sees unit-scale inputs
    x = make_batch(0, 1, 2)["hit"].x
    norm = model.tpc_encoder.input_norm.norm
    with torch.no_grad():
        norm["mean"].copy_(x.mean(dim=0))
        norm["var"].copy_(x.var(dim=0))
        norm["count"].fill_(x.size(0))
    model.tpc_encoder.input_norm.update = False
    return model


def evaluate(model: NuGraph3, *seeds: int) -> tuple[torch.Tensor, dict, Batch]:
    """Run one evaluation forward pass on a fresh batch of the given events"""
    batch = make_batch(*seeds)
    model.eval()
    with torch.no_grad():
        loss, metrics = model(batch, stage="val")
    return loss, metrics, batch


def test_predictions_independent_of_batch_mates():
    """An event's hit predictions must not depend on the other events in its batch"""
    model = build_model()
    _, _, alone = evaluate(model, 0)
    _, _, paired = evaluate(model, 0, 1)
    n = alone["hit"].num_nodes
    torch.testing.assert_close(paired["hit"].x_semantic[:n], alone["hit"].x_semantic)
    torch.testing.assert_close(paired["hit"].x_filter[:n], alone["hit"].x_filter)


DUPLICATE_INVARIANT = [
    "semantic/loss-val",
    "filter/loss-val",
    "event/loss-val",
    "vertex/loss-val",
    "spacepoint/loss-val",
    "instance/bkg-loss-val",
    "instance/potential-loss-val",
]


@pytest.mark.parametrize("key", DUPLICATE_INVARIANT)
def test_loss_invariant_to_duplicating_event(key):
    """A batch holding the same event twice must give the same per-event loss"""
    model = build_model()
    _, single, _ = evaluate(model, 0)
    _, double, _ = evaluate(model, 0, 0)
    torch.testing.assert_close(double[key], single[key], rtol=1e-4, atol=1e-6)


def test_particle_loss_invariant_to_batch_size():
    """The OC particle-loss term must not shrink as the batch grows"""
    model = build_model(particle_loss=True)
    _, single, _ = evaluate(model, 0)
    _, double, _ = evaluate(model, 0, 0)
    torch.testing.assert_close(double["instance/particle-loss-val"],
                               single["instance/particle-loss-val"], rtol=1e-4, atol=1e-6)


def test_potential_repels_within_event_only():
    """Hits are repelled by other particles in their own event, not in other events"""
    x = torch.tensor([[0.0], [0.5], [0.2], [0.7]]) # events [0, 0, 1, 1], one hit per particle
    f = torch.full((4,), 0.5)
    idx = torch.arange(4)
    v = ObjConLoss().l_v(x, f, idx, idx, idx, 4, torch.tensor([0, 0, 1, 1]))
    q = torch.tensor(0.5).atanh().square() + 0.5
    torch.testing.assert_close(v, 0.5 * q.square())


@bug("OC loss", "a true particle with no hits makes scatter_max return an "
        "out-of-range index")
def test_obj_con_loss_tolerates_particle_without_hits():
    """Particles with no hits (filter_true_particles=False) must not crash the loss"""
    torch.manual_seed(0)
    n_hit = 12
    x = torch.randn(n_hit, 8)
    f = torch.rand(n_hit) * 0.9
    y_i = torch.tensor([0] * 5 + [1] * 5 + [-1] * 2)
    e_true = torch.stack((torch.arange(10), y_i[:10]), dim=0)
    loss = ObjConLoss()(x, f, y_i, y_i.clone(), 3, e_true, None) # particle 2 has no hits
    assert torch.isfinite(loss).all()


def test_training_step_backward_bf16_autocast():
    """A full forward and backward pass must work under bf16 autocast"""
    model = build_model()
    model.train()
    batch = make_batch(0, 1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = model(batch, stage="train")
    loss.backward()
    assert torch.isfinite(loss)


def clustered_hits(n_graphs: int, semantic_index: int) -> NuGraphData | Batch:
    """Graphs whose hits sit in three tight clusters in the OC latent space"""
    graphs = []
    for g in range(n_graphs):
        torch.manual_seed(g)
        centres = torch.tensor([[0.] * 8, [5.] * 8, [-5.] * 8])
        ox = (centres.repeat_interleave(10, dim=0) + 0.01 * torch.randn(30, 8))
        data = NuGraphData()
        h = data["hit"]
        h.x = torch.zeros(30, 4)
        h.ox = ox
        h.of = torch.full((30,), 0.5)
        h.x_filter = torch.ones(30)
        h.x_semantic = torch.nn.functional.one_hot(
            torch.full((30,), semantic_index), len(SEMANTIC_CLASSES)).float()
        h.y_semantic = torch.zeros(30, dtype=torch.long)
        graphs.append(data)
    return Batch.from_data_list(graphs) if n_graphs > 1 else graphs[0]


def instance_decoder() -> InstanceDecoder:
    return InstanceDecoder(beta_features=32, coord_features=128, instance_features=8,
                           semantic_classes=SEMANTIC_CLASSES)


def test_materialize_finds_clusters():
    """Three well-separated clusters of signal hits become three particles"""
    data = clustered_hits(1, SEMANTIC_CLASSES.index("MIP"))
    instance_decoder().materialize(data)
    assert data["particle"].x.size(0) == 3


@bug(4, "diffuse class is hard-coded as index 6; with 5 classes the mask does nothing")
def test_materialize_ignores_diffuse_hits():
    """Hits predicted as diffuse must not be clustered into particles"""
    data = clustered_hits(1, SEMANTIC_CLASSES.index("diffuse"))
    instance_decoder().materialize(data)
    assert data["particle"].x.size(0) == 0


@bug(5, "particle batch vector is built with torch.full((0,), i) and is always empty")
def test_materialize_batch_vector():
    """Each predicted particle must record which graph in the batch it came from"""
    data = clustered_hits(2, SEMANTIC_CLASSES.index("MIP"))
    instance_decoder().materialize(data)
    torch.testing.assert_close(data["particle"].batch,
                               torch.tensor([0, 0, 0, 1, 1, 1]))


@bug(9, "the update flag is a plain attribute, so a resumed run starts updating again")
def test_input_norm_freeze_survives_checkpoint():
    """Freezing the input norm must survive a save and reload"""
    norm = InputNorm(3)
    norm.update = False
    restored = InputNorm(3)
    restored.load_state_dict(norm.state_dict())
    assert restored.update is False


@bug(9, "running stats are Parameters, so DDP never broadcasts them and "
        "the optimiser receives them")
def test_input_norm_stats_are_buffers():
    """Running normalization statistics must be buffers, not parameters"""
    assert not list(InputNorm(3).parameters())


def load_script(name: str):
    """Import a script from scripts/ as a module"""
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StopTraining(Exception):
    """Raised by the fake Trainer once its arguments are recorded"""


def trainer_kwargs(monkeypatch, tmp_path, argv: list[str]) -> dict:
    """Run scripts/train.py up to Trainer construction and return its kwargs"""
    train = load_script("train")
    monkeypatch.setenv("NUGRAPH_LOG", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["train.py", "--name", "test"] + argv)
    args, _ = train.configure()

    class FakeData: # pylint: disable=too-few-public-methods
        def __init__(self, *_, **__):
            pass

    class FakeModel: # pylint: disable=too-few-public-methods
        @classmethod
        def from_args(cls, *_):
            return cls()

    recorded = {}
    def fake_trainer(**kwargs):
        recorded.update(kwargs)
        raise StopTraining
    monkeypatch.setattr(train, "Data", FakeData)
    monkeypatch.setattr(train.pl, "Trainer", fake_trainer)
    with pytest.raises(StopTraining):
        train.train(args, FakeModel)
    return recorded


@bug(1, "--precision is parsed but never passed to pl.Trainer")
def test_train_passes_precision_to_trainer(monkeypatch, tmp_path):
    kwargs = trainer_kwargs(monkeypatch, tmp_path, ["--precision", "bf16-mixed"])
    assert kwargs.get("precision") == "bf16-mixed"


@bug(1, "V100 GPUs (Heimdall) need fp16; '16-mixed' is not an allowed --precision")
def test_train_accepts_fp16_precision(monkeypatch, tmp_path):
    kwargs = trainer_kwargs(monkeypatch, tmp_path, ["--precision", "16-mixed"])
    assert kwargs.get("precision") == "16-mixed"


@bug("BalanceSampler", "with balance_frac=0 the loop variable idx is never defined")
def test_balance_sampler_without_outliers():
    sampler = BalanceSampler(datasize=list(range(40)), batch_size=8, balance_frac=0.)
    assert sorted(iter(sampler)) == list(range(40))


@bug("BalanceSampler", "__len__ returns the dataset size, not the number of indices yielded")
def test_balance_sampler_length_matches_iteration():
    sampler = BalanceSampler(datasize=list(range(45)), batch_size=8, balance_frac=0.1)
    assert len(sampler) == len(list(iter(sampler)))


@bug("FeatureExtension", "node degree comes from unique() counts, which skip nodes with no edges")
def test_feature_extension_degree_with_isolated_node():
    """A hit with no planar edges gets degree 0 and does not shift other hits"""
    data = NuGraphData()
    h = data["hit"]
    h.pos = torch.tensor([[0., 0.], [1., 0.], [2., 1.], [3., 0.], [4., 1.]])
    h.x = torch.zeros(5, 2)
    h.plane = torch.zeros(5, dtype=torch.long)
    chain = torch.tensor([[1, 2, 3], [2, 3, 4]])
    data["hit", "delaunay-planar", "hit"].edge_index = torch.cat((chain, chain.flip(0)), dim=1)
    out = FeatureExtension(planes=("u",))(data)
    log_degree = out["hit"].x[:, -2]
    expected = torch.tensor([1., 1., 2., 2., 1.]).log() # node 0 clamps 0 -> 1
    torch.testing.assert_close(log_degree, expected)
