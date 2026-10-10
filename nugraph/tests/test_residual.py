"""Tests for residual connections in NuGraph message-passing blocks"""
import pytest
import torch

from nugraph.models.nugraph3.core import NuGraphBlock, NuGraphGeometryBlock


@pytest.mark.parametrize("bipartite", [False, True])
def test_residual_block_starts_as_identity(bipartite):
    """A residual block's update is initialised to zero, so it passes target
    features through unchanged until it is trained"""
    torch.manual_seed(0)
    n_src, f = (7, 6) if bipartite else (5, 6)
    block = NuGraphBlock(f, f, f, residual=True)
    x_dst = torch.randn(5, f)
    x = (torch.randn(n_src, f), x_dst) if bipartite else x_dst
    edge_index = torch.stack((torch.arange(n_src), torch.arange(n_src) % 5))
    torch.testing.assert_close(block(x, edge_index), x_dst)


def test_residual_block_learns():
    """Gradients reach both the update and the attention weights"""
    torch.manual_seed(0)
    block = NuGraphGeometryBlock(4, residual=True)
    x, pos = torch.randn(6, 4), torch.randn(6, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    block(x, pos, edge_index).square().sum().backward()
    assert block.net[-1].weight.grad.abs().sum() > 0
    assert len(block.net) == 3


def test_residual_needs_matching_features():
    """A residual block cannot change the target feature count"""
    with pytest.raises(ValueError):
        NuGraphBlock(4, 4, 8, residual=True)


def test_default_block_unchanged():
    """Without residual connections the block keeps its final activation"""
    block = NuGraphBlock(4, 4, 4)
    assert len(block.net) == 4 and not block.residual
