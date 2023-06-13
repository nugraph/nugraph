from argparse import ArgumentParser
import warnings

import h5py

from torch import tensor, cat
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from pytorch_lightning import LightningDataModule

from ..data import H5Dataset
from ..util import PositionFeatures, FeatureNormMetric, FeatureNorm

class H5DataModule(LightningDataModule):
    """PyTorch Lightning data module for neutrino graph data."""
    def __init__(self,
                 data_path: str,
                 batch_size: int,
                 prepare: bool = False):
        super().__init__()

        # for this HDF5 dataloader, worker processes slow things down
        # so we silence PyTorch Lightning's warnings
        warnings.filterwarnings("ignore", ".*does not have many workers.*")

        self.filename = data_path
        self.batch_size = batch_size

        if prepare:
            self.generate_samples()

        with h5py.File(self.filename) as f:
            try:
                self.planes = f['planes'].asstr()[()].tolist()
                self.classes = f['classes'].asstr()[()].tolist()
                train_samples = f['samples/train'].asstr()[()]
                val_samples = f['samples/validation'].asstr()[()]
                test_samples = f['samples/validation'].asstr()[()]
            except:
                print('samples not found in file! pass "prepare=True" to generate them')
                exit

        transform = Compose((PositionFeatures(self.planes), self.load_norm()))

        self.train_dataset = H5Dataset(self.filename, train_samples, transform)
        self.val_dataset = H5Dataset(self.filename, val_samples, transform)
        self.test_dataset = H5Dataset(self.filename, test_samples, transform)

    def generate_norm(self):
        with h5py.File(self.filename, 'r+') as f:

            metrics = None
            samples = list(f['dataset'].keys())
            loader = DataLoader(H5Dataset(self.filename, samples=samples),
                                batch_size=self.batch_size)

            print('  generating feature norm...')
            metrics = None
            for batch in tqdm.tqdm(loader):
                for p in self.planes:
                    x = cat([batch[p].pos, batch[p].x], dim=-1)
                    if not metrics:
                        num_feats = x.shape[-1]
                        metrics = { p: FeatureNormMetric(num_feats) for p in self.planes }
                    metrics[p].update(x)
            for p in self.planes:
                f[f'norm/{p}'] = metrics[p].compute()

    def load_norm(self):
        norm = {}
        try:
            with h5py.File(self.filename) as f:
                for p in self.planes:
                    norm[p] = tensor(f[f'norm/{p}'][()])
        except:
            print('feature normalisations not found in file! run generate_norm() to generate them.')
            exit
        return FeatureNorm(self.planes, norm)

    def generate_samples(self):
        with h5py.File(self.filename, 'r+') as f:
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          drop_last=True, shuffle=True, pin_memory=True)

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