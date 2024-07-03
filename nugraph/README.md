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

As an example, to train the network for semantic segmentation on the Heimdall cluster, one might run

```
scripts/train.py --data-path /raid/uboone/NuGraph2/NG2-paper.gnn.h5 \
                 --logdir /raid/$USER/logs --name default --version semantic-filter \
                 --semantic --filter
```

This command would start a network training using the requested input dataset, training with the semantic head enabled, and writing network parameters and metrics to the directory `/raid/$USER/logs/default/semantic-filter`.

### Training on SLURM clusters

If you're working on a cluster that uses the SLURM batch submission system, such as the Wilson cluster at Fermilab, then you'll need to submit training via a batch script instead. An example batch script `train_batch.sh` is included in the `scripts` subdirectory. If you're training on the Wilson cluster, you can submit a training job by running
```
sbatch scripts/train_batch.sh <args>
```
where `<args>` are the same argument you'd pass if you were executing the training script locally.

If you're training on a SLURM environment other than the Wilson cluster, you'll need to edit the SLURM directives in the script appropriately for the cluster you're working on before submitting.

### Metric logging

In the above example, training outputs including logging metrics would be written to a subdirectory of `/raid/$USER/logs`. We can use the Tensorboard interface to visualise these metrics and track the network's training progress. You can start Tensorboard using the following command:

```
tensorboard --port XXXX --bind_all --logdir /raid/$USER/logs --samples_per_plugin 'images=200'
```

In the above example, you should replace `XXXX` with a unique port number of your choosing. Provided you're forwarding that port when working over SSH, you can then access the interface in a local browser at `localhost:XXXX`.
