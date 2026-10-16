"""Tests for the NuGraph data module"""
import h5py
import torch

from nugraph.data import H5DataModule

from synthetic import PLANES, make_event


def splits(path, seed: int) -> dict[str, list[str]]:
    """Draw the sample splits for a file and read them back"""
    H5DataModule.generate_samples(str(path), seed)
    with h5py.File(path) as f:
        return {key: f[f"samples/{key}"].asstr()[()].tolist()
                for key in ("train", "validation", "test")}


def test_generate_samples_is_seeded(tmp_path):
    """The same seed always gives the same split; another seed gives another"""
    path = tmp_path / "graphs.h5"
    with h5py.File(path, "w") as f:
        for i in range(40):
            make_event(i).save(f, f"dataset/r0_sr0_evt{i}")
        f["planes"] = list(PLANES)
        f["gen"] = torch.tensor([2])
    first = splits(path, 0)
    assert [len(v) for v in first.values()] == [36, 2, 2]
    assert sorted(sum(first.values(), [])) == sorted(f"r0_sr0_evt{i}" for i in range(40))
    assert splits(path, 0) == first
    assert splits(path, 1) != first
