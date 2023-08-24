import torch
from torch.utils.data.sampler import Sampler
import random
import numpy as np
import sys
import math
import heapq
from random import randrange
np.set_printoptions(threshold=sys.maxsize)

class BalanceSampler(Sampler):
    def __init__(self, data_source, sample_sizes, batch_size, dset_frac):
        self.data_source = data_source
        self.sample_sizes = list(sample_sizes)
        self.batch_size = batch_size
        self.dset_frac = dset_frac

    def __iter__(self):
        # Retrieve dataset size
        dset_len = len(self.data_source)
        # Calculate number of batches in dataset
        num_batches = math.floor(dset_len / self.batch_size)

        # Assign N as a fraction of the dataset length
        num_outliers = int(math.floor(dset_len * self.dset_frac))
        if num_outliers > dset_len:
            print('Number of outliers is greater than dataset size')
            sys.exit()

        # Find the largest N values where N = num_outliers
        n_largest = heapq.nlargest(num_outliers, self.sample_sizes)

        sample_indices = []
        sorted_sample_sizes = sorted(self.sample_sizes)
        for size in sorted_sample_sizes:
            index = self.sample_sizes.index(size)
            sample_indices.append(index)

        # Separate the indices of the samples with the largest sizes
        if num_outliers == 0:
            n_largest_indices = []
        else:
            n_largest_indices = sample_indices[-num_outliers:]
        sample_indices = sample_indices[:dset_len-num_outliers]

        # Shuffle indices of n-largest values
        np.random.shuffle(n_largest_indices)
        np.random.shuffle(sample_indices)

        # Create as many bins as the number of batches
        bins = []
        for i in range(num_batches):
            bins.append([])

        # Distribute the n-largest sample indices to each bin
        for i, sample_index in enumerate(n_largest_indices):
            bin_index = i % self.batch_size
            bins[bin_index].append(sample_index)

        # Distribute the remaining samples to each bin
        #sample_indices = list(reversed(sample_indices))
        for i, sample_index in enumerate(sample_indices):
            bin_index = i % self.batch_size
            bins[bin_index].append(sample_index)

        # Shuffle each bin and append to indices array
        indices = []
        for bin in bins:
            np.random.shuffle(bin)
            indices += bin

        return iter(indices)

    def __len__(self):
        return len(self.data_source)
