import sys
import torch
import numpy as np 
import math
from torch_geometric.data import Data, Dataset, HeteroData

class SampleSizes():
    def __init__(self, f, samples):
        self.f = f
        self.samples = samples
        self.sample_sizes = self.generate_sample_sizes()

    def calc_sample_size(self, sample, dataset_names):
        sample_size = 0
        for dataset in dataset_names:
            store, attr = dataset.split('/')
            if "_" in store: store = tuple(store.split("_"))
            temp = sample[store][attr]
            temp_size = temp.nelement() * temp.element_size()
            #print('Number of elements: ', temp.nelement(),
            #      'Element size: ', temp.element_size())
            sample_size += temp_size
        #print('Sample size: ', sample_size)
        return sample_size

    def load_heterodata(self, f, name: str) -> HeteroData:
        data = HeteroData()
        # Read the whole dataset idx, dataset name is self.groups[idx]
        group = f[f'dataset/{name}'][()]
        dataset_names = group.dtype.names

        for dataset in dataset_names:
            store, attr = dataset.split('/')
            if "_" in store: store = tuple(store.split("_"))
            if attr in ['run','subrun','event','num_nodes']: # scalar
                data[store][attr] = torch.as_tensor(group[dataset][()])
            elif group[dataset].ndim == 0:
                # other zero-dimensional size datasets
                data[store][attr] = torch.LongTensor([[],[]])
            else: # multi-dimension array
                data[store][attr] = torch.as_tensor(group[dataset][:])
        return data, dataset_names

    def generate_sample_sizes(self):
        sample_sizes = []
        for sample in self.samples:
            processed_sample, dataset_names = self.load_heterodata(self.f, sample)
            sample_size = self.calc_sample_size(processed_sample, dataset_names)
            sample_sizes.append(sample_size)
        return sample_sizes