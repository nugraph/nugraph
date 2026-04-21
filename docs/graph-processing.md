# Graph processing

Event HDF5 files contain tables of low-level information on physics events, such as detector hits, true energy depositions and simulated particles. The processing stage is responsible for assembling that low-level information into nugraph-structured graph objects, using truth information to append training labels to those graph objects, and then storing the processed graphs into an output HDF5 file.

The `pynuml` package can be used to preprocess events from an event HDF5 file into a graph HDF5 file containing graph objects. If you do not have access to an event HDF5 file, please refer to previous documentation on [creating an event HDF5 file](create-hdf5) from an artroot dataset.

## Validating

The [processing notebook](https://github.com/nugraph/nugraph/blob/main/notebooks/process.ipynb) can be used to interactively visualize and validate graph processing interactively on individual events before being run at scale. This notebook initializes an input file, initializes some truth labelling algorithms, and then constructs a `HitGraphProducer` object to perform the processing. It then initializes a plotting utility, and steps through events interactively, invoking the processor to generate graphs and then visualizing them directly. By stepping through events in this manner and inspecting the processed graphs, one can easily validate the output and catch potential issues with graph processing before proceeding with the computationally expensive step of processing a graph dataset at scale.

## Processing

Running graph processing in `pynuml` is typically performed via an MPI parallel process, ideally on an HPC cluster with high-performance CPU cores. It can be performed serially without MPI, but is generally very slow to execute unless the underlying dataset is small. A simple processing script is provided at `scripts/process.py` in the nugraph repository ([link](https://github.com/nugraph/nugraph/blob/main/scripts/process.py)). This processing script provides a couple of useful configuration options, but the user is strongly encouraged to use this script as a template for a custom processing script depending on their needs. One may note the contents of the processing script are similar to those of the processing notebook above, which can be used to experiment and settle on a processing configuration for inside the processing script.

To invoke the processing script in parallel, simply run

```bash
mpiexec -n <N> process.py -i <input file> -o <output stem>
```

where `<N>` is the number of processes to parallelize over, `<input file>` is the name of the input event HDF5 file, and `<output stem>` is the stem of the output graph HDF5 file.

```{admonition} Note on output file naming
:class: warning

Note that output files will be written as `<output_stem>.XXXX.h5`, where `XXXX` is the number of the parallel process. It's best to omit the file extension `.h5` from this output stem, otherwise they'll be written as `output.h5.XXXX.h5`, which is ultimately fine but a little confusing.
```

## Labelling

The `HitGraphProducer` class accepts optional `semantic_labeller` and `event_labeller` arguments which allow the user to select truth labelling arguments. The `StandardLabels` class is the appropriate semantic labelling tool for most use cases. Depending on the intended physics outcomes, the event labelling algorithm may be the `FlavorLabels` class labelling different neutrino flavours, the `PDKLabels` class labelling events as either proton decay or atmospheric neutrino, or a totally separate labelling class defined by the user at runtime.

## Merging

Once parallel processing has been run, the final step before model training is to merge the processed graph HDF5 files. This can be accomplished by running the `merge.py` script provided [here](https://github.com/nugraph/nugraph/blob/main/scripts/merge.py), which takes the output stem from the previous step as input, merges together all files matching that pattern, writes a single merged output file and then removes the input files. It should be run like

```bash
python merge.py -f <output stem>
```

where `<output stem>` is the same output stem provided when running the processing script described above.

## Batch processing

An example SLURM batch processing script is provided at `scripts/train_batch.sh` [here](https://github.com/nugraph/nugraph/blob/main/scripts/process_batch.sh). This batch script performs the graph processing script in parallel with MPI, automatically scaling the number of parallel processes to match the number of SLURM processes requested, and then merges the processed file into a single merged graph file, all in a single batch submission. This method provides the most efficient and convenient method of graph processing, and is highly recommended if a suitable SLURM cluster is available.

This script can be submitted to a SLURM batch queue simply by running

```bash
sbatch process_batch.sh <input file> <output stem>
```

where `<input file>` is the input event HDF5 file and `<output stem>` is the stem of the output graph HDF5 file.

```{admonition} Note on batch scripts
:class: warning

The `process_batch.sh` script in the repository was designed for the Fermilab Wilson cluster, which has now been decommissioned. It should serve as a convenient template for submitting batch processing scripts at other sites, but the SLURM directives at the top of the script will need to be modified depending on the SLURM cluster being used.
```
