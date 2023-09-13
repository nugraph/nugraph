# NuGraph: a Graph Neural Network (GNN) for neutrino physics event reconstruction

This repository contains a GNN architecture for reconstructing particle interactions in neutrino physics detector environments. Its primary function is the classification of detector hit particle type through semantic segmentation, with additional secondary functions such as background hit rejection, event classification, clustering and vertex reconstruction.

## Installation

This repository can be installed in Python via `pip`, although using Anaconda to install dependencies is strongly recommended. Detailed instructions on how to easily install all necessary dependencies are available [here](https://pynuml.readthedocs.io/en/latest/install/installation.html).

Once dependencies are installed, you can simply clone this repository and installing it via `pip` – if you intend to carry out any development on the code, installing in editable mode is recommended:

```
git clone git@github.com:exatrkx/NuGraph
pip install --no-deps -e ./NuGraph
```

## Training a model

You can train the model using a processed graph dataset as input by executing the `train.py` script in the `scripts` subdirectory. This script accepts many arguments to configure your training – for a complete summary of all available arguments, you can simply run

```
scripts/train.py --help
```

As an example, to train the network for semantic segmentation on the Wilson cluster at Fermilab one might run

```
scripts/train.py --data-path /wclustre/fwk/exatrkx/data/uboone/CHEP2023/enhanced-vertex.gnn.h5 \
                 --semantic --logdir /work1/fwk/$USER/logs --name semantic \
```

This command would start a network training using the requested input dataset, training with the semantic head enabled, and writing network parameters and metrics to a subdirectory inside `/work1/fwk/$USER/logs/semantic`.

### Metric logging

In the above example, training outputs including logging metrics would be written to a subdirectory of `/work1/fwk/$USER/logs`. We can use the Tensorboard interface to visualise these metrics and track the network's training progress. You can start Tensorboard using the following command:

```
tensorboard --port XXXX --bind_all --logdir /work1/fwk/$USER/logs --samples_per_plugin 'images=200'
```

In the above example, you should replace `XXXX` with a unique port number of your choosing. Provided you're forwarding that port when working over SSH, you can then access the interface in a local browser at `localhost:XXXX`.