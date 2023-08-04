import torch
from torch.utils.data.sampler import Sampler
import random
import numpy as np
import sys
import math
from random import randrange
np.set_printoptions(threshold=sys.maxsize)

class BalanceSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        #self.outlier_num = 50

    def __iter__(self):
        dataset_len = len(self.data_source)
        indices = np.arange(dataset_len)
        
        # Balance batch size in bytes with bin partitioning 
        bins = []
        bin_size = math.floor(dataset_len/self.batch_size)
        for i in range(self.batch_size):
            start_idx = i * bin_size
            end_idx = (i+1) * bin_size
            curr_bin = indices[start_idx:end_idx]
            if i == self.batch_size - 1:
                curr_bin = indices[start_idx:]
            bins.append(curr_bin)

        bal_indices = []
        num_iters = int(np.ceil(dataset_len/self.batch_size))
        for i in range(num_iters):
            for j in range(self.batch_size):
                curr_bin = bins[j]
                bin_idx = randrange(len(curr_bin))
                dataset_idx = curr_bin[bin_idx]
                bal_indices.append(dataset_idx)
        
        return iter(indices)

    def __len__(self):
        return len(self.data_source)
