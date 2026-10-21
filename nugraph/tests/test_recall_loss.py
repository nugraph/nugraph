"""Tests for the recall loss"""
import torch
from torchmetrics.classification import MulticlassRecall

from nugraph.util import RecallLoss


def test_recall_loss_weights_match_multiclass_recall():
    """Class weights are one minus MulticlassRecall's per-class recall, with
    ignored labels left out and absent classes weighted 1"""
    torch.manual_seed(0)
    x = torch.randn(200, 5)
    y = torch.randint(0, 4, (200,))   # class 4 absent
    y[::7] = -1
    recall = MulticlassRecall(num_classes=5, average="none", ignore_index=-1)(x, y)
    ce = torch.nn.functional.cross_entropy(x, y, reduction="none", ignore_index=-1)
    expected = ((1 - recall)[y] * ce).sum() / (y != -1).sum()
    torch.testing.assert_close(RecallLoss(num_classes=5)(x, y), expected)
