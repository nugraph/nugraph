import torch
from torch.utils.data.sampler import Sampler
import numpy as np 
import math

class OutlierSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.outlier_num = 50

    def __iter__(self):
        dset_len = len(self.data_source)
        indices = np.arange(dset_len)
        outlier_indices = indices[dset_len-self.outlier_num:dset_len]
        indices = indices[:dset_len-self.outlier_num]
        np.random.shuffle(indices)

        print('Outlier indices: ', outlier_indices)
        for i, outlier in enumerate(outlier_indices):
            index = (i * self.batch_size) % dset_len
            indices = np.insert(indices, index, outlier)

        print('Indices: ', indices)
        return iter(indices)

    def __len__(self):
        return len(self.data_source)
