"""nugraph.data submodule"""
from .dataset import NuGraphDataset
from .data_module import NuGraphDataModule

# legacy imports
from .dataset import NuGraphDataset as H5Dataset
from .data_module import NuGraphDataModule as H5DataModule
