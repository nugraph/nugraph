# nugraph documentation

**nugraph** is a graph neural network (GNN) providing particle reconstruction for neutrino physics experiments, supported by a software ecosystem for graph processing, truth labelling and network training and inference. The **pynuml** package provides an HDF5 IO interface for efficiently preprocessing large datasets in parallel with MPI, as well as various algorithms for particle labelling, and tools for graph visualization, while the **nugraph** package provides the NuGraph2 and NuGraph3 model architectures in addition to a custom Lightning data module for efficiently loading graph objects from file.

```{toctree}
:maxdepth: 1
:caption: Getting Started

installation
create-hdf5
graph-processing
training-and-inference
```
