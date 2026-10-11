"""Tests for skipping batches that do not fit in GPU memory"""
import pytest
import torch
from pytorch_lightning import LightningModule, Trainer

from nugraph.data import H5DataModule
from test_nexus_decoder import nexus_batch, nexus_model


def fit(model, batches: int) -> Trainer:
    """Train for one epoch on synthetic batches, on the CPU"""
    data = [nexus_batch(i, i + 1) for i in range(batches)]
    loader = torch.utils.data.DataLoader(data, batch_size=None, collate_fn=lambda b: b)
    trainer = Trainer(accelerator="cpu", max_epochs=1, logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False, log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=loader)
    return trainer


def test_out_of_memory_batches_are_skipped(monkeypatch):
    """A batch that runs out of memory in the forward or the backward pass is
    skipped, leaving the trained parameters as they were (running input
    statistics still update in the forward pass), and training goes on"""
    model = nexus_model()
    forward, backward = model.forward, LightningModule.backward
    calls = {"forward": 0, "backward": 0}
    def oom_forward(*args, **kwargs):
        calls["forward"] += 1
        if calls["forward"] == 2:   # batch 1
            raise torch.OutOfMemoryError("test")
        return forward(*args, **kwargs)
    def oom_backward(self, *args, **kwargs):
        calls["backward"] += 1
        if calls["backward"] == 3:  # batch 3 (batch 1 has no backward pass)
            raise torch.OutOfMemoryError("test")
        return backward(self, *args, **kwargs)
    monkeypatch.setattr(model, "forward", oom_forward)
    monkeypatch.setattr(LightningModule, "backward", oom_backward)
    params = {}
    def snapshot(when):
        return lambda batch, batch_idx, *a: params.setdefault((when, batch_idx),
            [p.detach().clone() for p in model.parameters() if p.requires_grad])
    monkeypatch.setattr(model, "on_train_batch_start", snapshot("start"))
    monkeypatch.setattr(model, "on_train_batch_end", lambda out, batch, batch_idx:
                        snapshot("end")(batch, batch_idx))
    with pytest.warns(UserWarning, match="CUDA out of memory"):
        fit(model, 5)
    assert model.oom_skipped == 2 and model.oom_in_a_row == 0
    def same(i):
        return all(torch.equal(a, b) for a, b in zip(params[("start", i)], params[("end", i)]))
    assert same(1) and same(3)
    assert not same(0) and not same(2) and not same(4)


def test_ten_out_of_memory_batches_in_a_row_end_training(monkeypatch):
    """Ten skipped batches in a row still end the run"""
    model = nexus_model()
    def oom_forward(*args, **kwargs):
        raise torch.OutOfMemoryError("test")
    monkeypatch.setattr(model, "forward", oom_forward)
    with pytest.raises(torch.OutOfMemoryError, match="10 batches in a row"):
        with pytest.warns(UserWarning):
            fit(model, 12)


def test_graph_build_out_of_memory_gives_none():
    """A batch whose spacepoint graph does not fit in GPU memory becomes None"""
    class Transform:
        def __call__(self, batch):
            raise torch.OutOfMemoryError("test")
    data = type("Data", (), {"sp_transform": Transform()})()
    assert H5DataModule.on_after_batch_transfer(data, nexus_batch(0), 0) is None
