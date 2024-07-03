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

from ..data import H5Dataset, BalanceSampler
from ..util import PositionFeatures, FeatureNormMetric, FeatureNorm, HierarchicalEdges, EventLabels

class H5DataModule(LightningDataModule):
    """PyTorch Lightning data module for neutrino graph data."""
    def __init__(self,
                 data_path: str,
                 batch_size: int,
                 shuffle: str = 'random',
                 balance_frac: float = 0.1,
                 prepare: bool = False):
        super().__init__()

        # for this HDF5 dataloader, worker processes slow things down
        # so we silence PyTorch Lightning's warnings
        warnings.filterwarnings("ignore", ".*does not have many workers.*")

        self.filename = data_path
        self.batch_size = batch_size
        if shuffle != 'random' and shuffle != 'balance':
            print('shuffle argument must be "random" or "balance".')
            sys.exit()
        self.shuffle = shuffle
        self.balance_frac = balance_frac

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
                test_samples = f['samples/test'].asstr()[()]
            except:
                print('Sample splits not found in file! Call "generate_samples" to create them.')
                sys.exit()

            # load data sizes
            try:
                self.train_datasize = f['datasize/train'][()]
            except:
                print('Data size array not found in file! Call "generate_samples" to create it.')
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
                             FeatureNorm(self.planes, norm),
                             HierarchicalEdges(self.planes),
                             EventLabels()))

        self.train_dataset = H5Dataset(self.filename, train_samples, transform)
        self.val_dataset = H5Dataset(self.filename, val_samples, transform)
        self.test_dataset = H5Dataset(self.filename, test_samples, transform)

    @staticmethod
    def generate_samples(data_path: str):
        with h5py.File(data_path) as f:
            samples = list(f['dataset'].keys())
        split = int(0.05 * len(samples))
        splits = [ len(samples)-(2*split), split, split ]
        train, val, test = random_split(samples, splits)

        with h5py.File(data_path, "r+") as f:
            for name in [ 'train', 'validation', 'test' ]:
                key = f'samples/{name}'
                if key in f:
                    del f[key]

        with h5py.File(data_path, "r+") as f:
            f.create_dataset("samples/train", data=list(train))
            f.create_dataset("samples/validation", data=list(val))
            f.create_dataset("samples/test", data=list(test))

        with h5py.File(data_path, "r+") as f:
            try:
                planes = f['planes'].asstr()[()].tolist()
            except:
                print('Metadata not found in file! "planes" is required.')
                sys.exit()

        with h5py.File(data_path, "r+") as f:
            if 'datasize/train' in f:
                del f['datasize/train']
        transform = PositionFeatures(planes)
        dataset = H5Dataset(data_path, train, transform)
        def datasize(data):
            ret = 0
            for store in data.stores:
                for val in store.values():
                    ret += val.element_size() * val.nelement()
            return ret
        dsize = [datasize(data) for data in tqdm.tqdm(dataset)]
        del dataset
        with h5py.File(data_path, "r+") as f:
            f.create_dataset('datasize/train', data=dsize)
            
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
                key = f'norm/{p}'
                if key in f:
                    del f[key]
                f[key] = metrics[p].compute()

    def train_dataloader(self) -> DataLoader:
        if self.shuffle == 'balance':
            shuffle = False
            sampler = BalanceSampler.BalanceSampler(
                        datasize=self.train_datasize,
                        batch_size=self.batch_size, 
                        balance_frac=self.balance_frac)
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
                          default='/raid/uboone/NuGraph2/NG2-paper.gnn.h5',
                          help='Location of input data file')
        data.add_argument('--batch-size', type=int, default=64,
                          help='Size of each batch of graphs')
        data.add_argument('--limit_train_batches', type=int, default=None,
                          help='Max number of training batches to be used')
        data.add_argument('--limit_val_batches', type=int, default=None,
                          help='Max number of validation batches to be used')
        data.add_argument('--shuffle', type=str, default='balance',
                          help='Dataset shuffling scheme to use')
        data.add_argument('--balance-frac', type=float, default=0.1,
                          help='Fraction of dataset to use for workload balancing')
        return parser
