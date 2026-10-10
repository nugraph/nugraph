"""Tests for NuGraph3 semantic feedback from the spacepoint stage to hits"""
import torch

from test_nexus_decoder import nexus_batch, nexus_model

E_SP = ("sp", "sp3d", "sp")


def test_feedback_starts_as_without():
    """The feedback update starts at zero, so hit predictions match a model without it"""
    without = nexus_model(nexus_semantic=True)
    model = nexus_model(nexus_semantic=True, nexus_feedback=True)
    without.eval()
    model.eval()
    batch = nexus_batch(0, 1)
    with torch.no_grad():
        a, b = batch.clone(), batch.clone()
        without(a)
        model(b)
    torch.testing.assert_close(a["hit"].x_semantic, b["hit"].x_semantic)


def test_feedback_training_step():
    """A training step trains the feedback update and reports one semantic pass"""
    model = nexus_model(nexus_semantic=True, nexus_feedback=True)
    model.train()
    loss, metrics = model(nexus_batch(0, 1), stage="train")
    loss.backward()
    assert torch.isfinite(loss)
    assert "semantic/recall-train" in metrics
    assert model.feedback.hit_net[-1].weight.grad.abs().sum() > 0


def test_feedback_does_not_train_scores():
    """Edge and ghost scores are only weights: no gradient flows back into them"""
    model = nexus_model(nexus_semantic=True, nexus_feedback=True)
    model.eval()
    batch = nexus_batch(0, 1)
    with torch.no_grad():
        model(batch)
        model.feedback.hit_net[-1].weight.normal_()
    score = batch[E_SP].x.clone().requires_grad_()
    real = batch["sp"].x_filter.clone().requires_grad_()
    batch[E_SP].x, batch["sp"].x_filter = score, real
    batch["hit"].x = batch["hit"].x.detach()
    model.feedback(batch)
    batch["hit"].x.sum().backward()
    assert score.grad is None and real.grad is None
