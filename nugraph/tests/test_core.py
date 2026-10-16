"""Tests for the NuGraph3 core message-passing block"""
import pytest
import torch

from nugraph.models.nugraph3.core import NuGraphBlock


@pytest.mark.parametrize("bipartite", [False, True])
def test_block_output_is_layer_normalized(bipartite):
    """Updated node features have zero mean and unit variance per node"""
    torch.manual_seed(0)
    n_src, f_src = (7, 6) if bipartite else (5, 4)
    n_dst, f_dst = 5, 4
    block = NuGraphBlock(f_src, f_dst, 16)
    x_dst = torch.randn(n_dst, f_dst)
    x = (torch.randn(n_src, f_src), x_dst) if bipartite else x_dst
    edge_index = torch.stack((torch.arange(n_src), torch.arange(n_src) % n_dst))
    out = block(x, edge_index)
    assert out.shape == (n_dst, 16)
    torch.testing.assert_close(out.mean(dim=1), torch.zeros(n_dst), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.var(dim=1, unbiased=False), torch.ones(n_dst),
                               atol=1e-2, rtol=0)
