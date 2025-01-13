:github_url: https://github.com/nugraph/nugraph

nugraph documentation
=====================

**pynuml** is a python package providing a data interface for machine learning in neutrino physics. It utilises the **NuML** HDF5 event file format to efficiently preprocess physics events into ML objects for training neural networks. It is designed to abstract away many aspects of a typical ML workflow:

- Efficiently iterate over large HDF5 datasets
- Generate semantic and instance labels for particles
- Preprocess events into ML objects

**nugraph** is a python package providing graph neural networks for neutrino physics event reconstruction. It offers tools for loading the graph objects produced by pynuml during model training, and two different generations of GNN architecture, NuGraph2 and NuGraph3.

.. toctree::
   :maxdepth: 1
   :caption: Installation

   installation

.. toctree::
   :maxdepth: 1
   :caption: pynuml documentation

   pynuml/io
   pynuml/data
   pynuml/labels
   pynuml/process
   pynuml/plot

.. toctree::
   :maxdepth: 1
   :caption: nugraph documentation

   nugraph/data
   nugraph/models
   nugraph/util