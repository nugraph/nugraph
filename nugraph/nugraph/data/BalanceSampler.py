import sys
import numpy as np
from torch.utils.data.sampler import Sampler

class BalanceSampler(Sampler):
    def __init__(self, datasize, batch_size, balance_frac):
        self.datasize = list(datasize)
        self.batch_size = batch_size
        self.balance_frac = balance_frac

    def __iter__(self):
        # Retrieve dataset size
        dset_len = len(self.datasize)
        # Calculate number of batches in dataset
        num_batches = int(np.floor(dset_len / self.batch_size))

        # Assign N as a fraction of the dataset length
        num_outliers = int(np.floor(dset_len * self.balance_frac))
        if num_outliers > dset_len:
            print('Number of outliers is greater than dataset size')
            sys.exit()

        sample_indices = np.argsort(self.datasize)

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
        bins = [ [] for i in range(num_batches) ]

        # Distribute the n-largest sample indices to each bin
        for i, sample_index in enumerate(n_largest_indices):
            idx = i % num_batches
            bins[idx].append(sample_index)

        # Distribute the remaining samples to each bin
        offset = idx + 1
        for i, sample_index in enumerate(sample_indices):
            idx = (i + offset) % num_batches
            # drop last batch
            if len(bins[idx]) == self.batch_size:
                break
            bins[idx].append(sample_index)

        # Shuffle each bin and append to indices array
        indices = []
        for bin in bins:
            np.random.shuffle(bin)
            indices += bin

        return iter(indices)

    def __len__(self):
        return len(self.datasize)