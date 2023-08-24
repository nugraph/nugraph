import torch
from torch.utils.data.sampler import Sampler
import numpy as np 
import math
import heapq

class SkewSampler(Sampler):
    def __init__(self, data_source, sample_sizes, batch_size):
        self.data_source = data_source
        self.sample_sizes = list(sample_sizes)
        self.batch_size = batch_size

    def __iter__(self):
        # Calculate number of batches in dataset
        dset_len = len(self.data_source)
        num_batches = math.floor(dset_len / self.batch_size)

        # Set the number of skewed values in the dataset
        num_skew = self.batch_size

        # Find the largest N values where N = num_skew
        n_largest = heapq.nlargest(num_skew, self.sample_sizes)

        # Find the indices of the samples with the largest sizes
        n_largest_indices = []
        for size in n_largest:
            index = self.sample_sizes.index(size)
            n_largest_indices.append(index)

        # Generate indices and remove the indices of the n-largest samples 
        if num_skew > dset_len:
            print('Number of skewed values is greater than dataset size')
            sys.exit()
        indices = []
        while len(indices) < dset_len:
            if len(n_largest_indices) + len(indices) < dset_len:
                np.random.shuffle(n_largest_indices)
                indices += n_largest_indices
            else:
                diff = dset_len - len(indices)
                np.random.shuffle(n_largest_indices)
                indices += n_largest_indices[:diff]

        return iter(indices)

    def __len__(self):
        return len(self.data_source)
