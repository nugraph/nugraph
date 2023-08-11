import torch
from torch.utils.data.sampler import Sampler
import numpy as np 
import math

class SkewSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.skew_num = 50

    def __iter__(self):
        dset_len = len(self.data_source)
        indices = np.arange(dset_len)
        # Fill with outliers only
        if self.skew_num < dset_len:
            div = int(math.floor(dset_len / self.skew_num))
            rem = dset_len % self.skew_num
            outlier_indices = indices[dset_len-self.skew_num:dset_len]
            for i in range(div):
                indices[i*self.skew_num:(i+1)*self.skew_num] = outlier_indices
            indices[-rem:]= outlier_indices[:rem]
        return iter(indices)

    def __len__(self):
        return len(self.data_source)
