"""Tests for the NuGraph HDF5 dataset"""
import pickle

import h5py
import torch
from torch_geometric.loader import DataLoader

from pynuml.data import NuGraphData
from nugraph.data import NuGraphDataset


def make_file(path, n: int = 8) -> list[str]:
    """Write n small graphs to an HDF5 file and return their names"""
    names = [f"evt{i}" for i in range(n)]
    with h5py.File(path, "w") as f:
        for i, name in enumerate(names):
            data = NuGraphData()
            data["hit"].x = torch.full((3, 2), float(i))
            data.save(f, f"dataset/{name}")
    return names


def test_dataset_pickles_after_reading(tmp_path):
    """A dataset that has read from its file can be pickled, and the copy reopens it"""
    names = make_file(tmp_path / "data.h5")
    dataset = NuGraphDataset(str(tmp_path / "data.h5"), names)
    expected = dataset[3]["hit"].x
    restored = pickle.loads(pickle.dumps(dataset))
    torch.testing.assert_close(restored[3]["hit"].x, expected)


def test_dataloader_workers_with_spawn(tmp_path):
    """Workers started with spawn load the same batches as the main process"""
    names = make_file(tmp_path / "data.h5")
    dataset = NuGraphDataset(str(tmp_path / "data.h5"), names)
    single = list(DataLoader(dataset, batch_size=2))
    workers = list(DataLoader(dataset, batch_size=2, num_workers=2,
                              multiprocessing_context="spawn"))
    assert len(workers) == len(single) == 4
    for a, b in zip(single, workers):
        torch.testing.assert_close(b["hit"].x, a["hit"].x)
