import torch
from torch.utils.data.sampler import Sampler
import numpy as np 
import math

class SkewSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.outlier_num = 50

    def __iter__(self):
        dset_len = len(self.data_source)
        indices = np.arange(dset_len)
        # Fill with outliers only
        div = int(math.floor(dset_len / self.batch_size))
        rem = dset_len % self.batch_size 
        outlier_indices = indices[dset_len-self.batch_size:dset_len]
        for i in range(div):
            indices[i*self.batch_size:(i+1)*self.batch_size] = outlier_indices
        indices[-rem:]= outlier_indices[:rem]
        #outliers = indices[dset_len - self.outlier_num:dset_len]
        return iter(indices)

    def __len__(self):
        return len(self.data_source)
