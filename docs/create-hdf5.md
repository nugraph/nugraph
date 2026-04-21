# Creating Event HDF5 files

## Installing the numl package

The primary input to the nugraph workflow is an Event HDF5 file containing tables of low-level information such as detector hits, true energy depositions and particles. Such files are typically created using the NuML LArSoft package, which is available [here](https://github.com/nugraph/numl). This repository can be installed in MRB as a LArSoft package in the standard manner, by running

```bash
mrb g https://github.com/nugraph/numl
```

and then built as a regular LArSoft package. Tags of the repository are available for a range of `dunesw` versions.

## Creating event HDF5 files

Once NuML is installed, Event HDF5 files can be created from artroot files by processing with LArSoft by running

```bash
lar -c hdf5maker_dune.fcl [artroot file]
```

which will summarize information from the Art event record into event HDF5 files.

## Merging event HDF5 files

The `pynuml` package is designed to efficiently preprocess large datasets using MPI. In order to do this, it is designed to run over a single large event HDF5 file. However, since most artroot datasets consist of a large number of files, most workflows will produce a small number of larger HDF5 files. For this reason, an additional merging step is performed to combine these smaller event HDF5 files into a single large file.

```{admonition} Note on run numbers
:class: warning

Both the event and graph HDF5 files use the run, subrun and event indices to uniquely identify each event. If you are generating simulation datasets across multiple artroot files, please ensure that the [run, subrun, event] indices in every file are unique and not repeated. Attempting to merge event HDF5 files containing the same event metadata indices will result in errors, undefined behaviour and malformed events.
```

Event files can be merged using the `ph5concat` utility. This utility is available on conda, and can be easily installed by running

```bash
conda install -c numl ph5concat
```

File merging can be parallized over N MPI processes by running

```bash
mpiexec -n <N> ph5_concat -k event_table/event_id -i <input.txt> -o <output.evt.h5>
```

where `<N>` is the number of parallel threads to use, `<input.txt>` is a text file containing the list of input HDF5 files to merge (one file per line), and `<output.evt.h5>` is the name of the merged output event HDF5 file. By convention, numl event HDF5 files are given the `evt.h5` extension.

Once the file is merged, additional sequencing metadata must be added before it can be used for graph processing. This metadata can be easily added using the `add_key` utility, which is included in the `ph5concat` package, by running the following:

```bash
add_key -f -k event_table/event_id -a <output.evt.h5>
```

This command will modify the merged event HDF5 file in place, appending additional datasets to the file containing sequencing metadata. Once a merged event HDF5 file has been created, and sequencing metadata added, it can then be used to [process graph datasets](graph-processing).
