from argparse import ArgumentParser
import warnings

import sys
import h5py
import tqdm

from torch import tensor, cat
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from pytorch_lightning import LightningDataModule

from ..data import H5Dataset
from ..data import SampleSizes
from ..util import PositionFeatures, FeatureNormMetric, FeatureNorm
from ..data import BalanceSampler

class H5DataModule(LightningDataModule):
    """PyTorch Lightning data module for neutrino graph data."""
    def __init__(self,
                 data_path: str,
                 batch_size: int,
                 shuffle_scheme: str = 'random',
                 dset_frac: float = 0.1,
                 prepare: bool = False):
        super().__init__()

        # for this HDF5 dataloader, worker processes slow things down
        # so we silence PyTorch Lightning's warnings
        warnings.filterwarnings("ignore", ".*does not have many workers.*")

        self.filename = data_path
        self.batch_size = batch_size
        self.shuffle_scheme = shuffle_scheme
        self.dset_frac = dset_frac

        with h5py.File(self.filename) as f:

            # load metadata
            try:
                self.planes = f['planes'].asstr()[()].tolist()
                self.semantic_classes = f['semantic_classes'].asstr()[()].tolist()
            except:
                print('Metadata not found in file! "planes" and "semantic_classes" are required.')
                sys.exit()

            # load optional event labels
            if 'event_classes' in f:
                self.event_classes = f['event_classes'].asstr()[()].tolist()
            else:
                self.event_classes = None

            # load sample splits
            try:
                train_samples = f['samples/train'].asstr()[()]
                val_samples = f['samples/validation'].asstr()[()]
                test_samples = f['samples/validation'].asstr()[()]
            except:
                print('Sample splits not found in file! Call "generate_samples" to create them.')
                sys.exit()

            # load sample sizes
            try:
                train_sample_sizes = f['sample_sizes/train'][()]
                val_sample_sizes = f['sample_sizes/val'][()]
                test_sample_sizes = f['sample_sizes/test'][()]
            except:
                print('Sample sizes not found in file! Call "generate_samples" to create them.')
                sys.exit()

            # load feature normalisations
            try:
                norm = {}
                for p in self.planes:
                    norm[p] = tensor(f[f'norm/{p}'][()])
            except:
                print('Feature normalisations not found in file! Call "generate_norm" to create them.')
                sys.exit()

        transform = Compose((PositionFeatures(self.planes),
                             FeatureNorm(self.planes, norm)))

        self.train_dataset = H5Dataset(self.filename, train_samples, transform)
        self.val_dataset = H5Dataset(self.filename, val_samples, transform)
        self.test_dataset = H5Dataset(self.filename, test_samples, transform)
        
        self.train_sample_sizes = train_sample_sizes
        self.val_sample_sizes = val_sample_sizes
        self.test_sample_sizes = test_sample_sizes

    @staticmethod
    def generate_samples(data_path: str):
        with h5py.File(data_path, 'r+') as f:
            samples = list(f['dataset'].keys())
            split = int(0.05 * len(samples))
            splits = [ len(samples)-(2*split), split, split ]
            train, val, test = random_split(samples, splits)

            for key in [ 'train', 'validation', 'test' ]:
                name = f'samples/{key}'
                if f.get(f'samples/{key}') is not None:
                    del f[f'samples/{key}']

            f['samples/train'] = list(train)
            f['samples/validation'] = list(val)
            f['samples/test'] = list(test)
            
            trainSampleSizes = SampleSizes.SampleSizes(f, list(train))
            train_sample_sizes = trainSampleSizes.sample_sizes

            valSampleSizes = SampleSizes.SampleSizes(f, list(val))
            val_sample_sizes = valSampleSizes.sample_sizes

            testSampleSizes = SampleSizes.SampleSizes(f, list(test))
            test_sample_sizes = testSampleSizes.sample_sizes

            for key in [ 'train', 'validation', 'test' ]:
                name = f'sample_sizes/{key}'
                if f.get(f'sample_sizes/{key}') is not None:
                    del f[f'sample_sizes/{key}']

            f['sample_sizes/train'] = train_sample_sizes
            f['sample_sizes/val'] = val_sample_sizes
            f['sample_sizes/test'] = test_sample_sizes
            
    @staticmethod
    def generate_norm(data_path: str, batch_size: int):
        with h5py.File(data_path, 'r+') as f:
            # load plane metadata
            try:
                planes = f['planes'].asstr()[()].tolist()
            except:
                print('Metadata not found in file! "planes" is required.')
                sys.exit()

            loader = DataLoader(H5Dataset(data_path,
                                          list(f['dataset'].keys()),
                                          PositionFeatures(planes)),
                                batch_size=batch_size)

            print('  generating feature norm...')
            metrics = None
            for batch in tqdm.tqdm(loader):
                for p in planes:
                    if not metrics:
                        num_feats = batch[p].x.shape[-1]
                        metrics = { p: FeatureNormMetric(num_feats) for p in planes }
                    metrics[p].update(batch[p].x)
            for p in planes:
                f[f'norm/{p}'] = metrics[p].compute()

    def train_dataloader(self) -> DataLoader:
        if self.shuffle_scheme == 'balance':
            shuffle = False
            sampler = BalanceSampler.BalanceSampler(
                        data_source=self.train_dataset,
                        sample_sizes=self.train_sample_sizes,
                        batch_size=self.batch_size, 
                        dset_frac=self.dset_frac)
        else:
            shuffle = True
            sampler = None

        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          sampler=sampler, drop_last=True, 
                          shuffle=shuffle, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size)

    @staticmethod
    def add_data_args(parser: ArgumentParser) -> ArgumentParser:
        data = parser.add_argument_group('data', 'Data module configuration')
        data.add_argument('--data-path', type=str,
                          default='/data/CHEP2023/filtered.gnn.h5',
                          help='Location of input data file')
        data.add_argument('--batch-size', type=int, default=64,
                          help='Size of each batch of graphs')
        data.add_argument('--limit_train_batches', type=int, default=None,
                          help='Max number of training batches to be used')
        data.add_argument('--limit_val_batches', type=int, default=None,
                          help='Max number of validation batches to be used')
        return parser