"""Tests for the balance sampler"""
import numpy as np

from nugraph.data.BalanceSampler import BalanceSampler


def test_balance_sampler_length_matches_indices():
    """The sampler's length is the number of indices it yields (whole batches
    only), which distributed training relies on to split them between GPUs"""
    np.random.seed(0)
    for size, batch_size in ((1000, 32), (997, 7), (64, 32)):
        sampler = BalanceSampler(np.random.rand(size), batch_size, 0.1)
        indices = list(sampler)
        assert len(sampler) == len(indices) == size // batch_size * batch_size
        assert len(set(indices)) == len(indices)
