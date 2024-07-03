# PyNuML: HDF5 IO and ML processing for neutrino physics

PyNuML is a Python toolkit for processing machine learning (ML) inputs from neutrino physics simulation datasets. It offers efficient MPI parallel processing of datasets, including standardised solutions for generating semantic and instance labels from low-level particle simulation, and constructing PyTorch ML inputs such as pixel maps and graphs. The package uses a modular design to maximise flexibility and extensibility, allowing the user to write custom labelling and/or object formation code in place of the algorithms provided.

## Parallel Event IO

HDF5 files produced using the [NuML standard](https://github.com/nugraph/numl) contain tabular data structures representing events, simulated particles, energy depositions, detector hits and any other information defined by the user. For large datasets, accessing the rows of a table corresponding to a specific event based on event index can become prohibitively slow. PyNuML includes a metadata standard for efficient MPI parallel IO with large-scale physics event data. This approach enables very efficient processing of datasets using MPI parallel processing on HPC nodes, while also providing a simple and effective interface for interactive analysis.

## Semantic and instance labelling

Novel ML techniques developed for particle physics typically conform to one of several standard archetypes: event classification, instance segmentation to cluster of detector hits into particles, or semantic segmentation of hits and/or particles into particle types. These applications typically utilise supervised learning, leveraging the detailed simulation already available to produce truth-labelled ML objects for model training.

Most of these experiments utilise the same primary workflow: primary particles from a generator are passed into Geant4 to simulate true energy depositions, which are in turn passed through detector simulation to produce simulated raw detector output. Generating ML truth labels for detector objects such as hits typically involves backtracking from detector-level information to access the underlying true particle information, and using that information to design some kind of instance label.

Many physicists producing ML inputs develop such a workflow from scratch, unnecessarily re-developing variants on the same basic mechanism over and over again, and often falling into the same pitfalls in the process. For instance, a user producing a CNN pixel map from detector hits will often loop over each hit, query a backtracker to fetch the associated true particle information, and then use that information to categorise that hit according to a user-defined semantic labelling scheme. This approach can become highly inefficient and convoluted as computational cycles are wasted re-categorising hits produced by the same simulated particle, especially if the labelling requires context information from parent or child particles.

PyNuML maximises efficiency by performing a single labelling pass over the true particle table, stepping hierarchically down from primary particles, assigning each particle a semantic and instance label using a standard taxonomy. These labels can then be efficiently propagated to detector objects using Pandas DataFrame merge operations, using the true energy deposition table as an intermediary. This also avoids double-counting errors that can occur when aggregating objects into pixel or voxel maps is necessary.

If the user's simulation includes custom Geant4 physics processes that necessitate modifications to a standard labelling scheme – or if they simply prefer a different labelling scheme altogether – the user can simply write their own labelling function to use instead. If the user develops a new labelling function that has general appeal, that function can then be added to the standard labelling options included in PyNuML.

## ML object formation

PyNuML also provides standard tools for the production of ML inputs, taking Pandas DataFrames containing event information and using them to construct a single ML input. A function that produces detector hit graphs for GNN training is provided, with 2D and 3D pixel map production in development. This single-event processing function is nested within an MPI parallel IO infrastructure to efficiently preprocess an entire dataset into ML inputs at scale, storing each object in an individual Pytorch `.pt` format or (experimentally) storing all inputs as compound data objects within a single HDF5 file.

## Getting started

### Dependencies

In order to correctly install all dependency packages, it's recommended to work within an Anaconda installation with NuML dependencies installed. If you don't already have Anaconda installed, we recommend using [Mambaforge](https://github.com/conda-forge/miniforge). A conda environment file is available via the anaconda client, and can be installed by running
```
mamba install -y anaconda-client
mamba env create numl/numl
```
This will install all dependencies necesssary for working with NuML. Once this environment is installed, it can be activated in a terminal session by running
```
mamba activate numl
```

This conda environment includes the latest release version of PyNuML, and all its dependencies.

### Installation

Although installation via the numl conda environment described above is strongly recommended, PyNuML can also be installed standalone. It can be installed via conda with
```
mamba install -c numl pynuml
```
and via pip with
```
pip install pynuml
```

For those who wish to contribute to developing PyNuML, or work with the current development build, the repository can be cloned and installed in editable mode using pip:
```
git clone https://github.com/nugraph/pynuml
pip install -e ./pynuml
```

If installed using this method, any modifications made to your local PyNuML release will be reflected in the `pynuml` module when imported at runtime. Again, using the numl conda environment is recommended -- for developers, the user can simply follow the instructions above to create a numl conda environment, and then override their conda installion of pynuml with the local repository by running
```
pip install --no-deps -e ./pynuml
```
