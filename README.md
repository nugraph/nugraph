# NuGraph

NuGraph is a graph neural network (GNN) for neutrino physics event reconstruction. This repository contains the source code for the following packages:
- [nugraph](nugraph/README.md) – module containing the GNN architecture and data loader
- [pynuml](pynuml/README.md) – module containing graph processing, truth labelling and visualization tools.

Two files uploaded
- nugraph/models/nugraph3/nugraph3.py  and   train.sh (for convenient training command)
- No changes in scripts/train.py --- it automatically picks up new Michel arguments from the model
- Use train.sh for running it. Users might need to make it executable first.
- Commands that can be used
- ./train.sh with_michel       -----  Michel physics regularization with 1% regularization applied
- ./train.sh without_michel    -----  Standard training without energy-based regularization
- ./train.sh strong_michel     -----  Strong Michel physics regularization with 10% regularization applied
- Users need to modify DATA_PATH in train.sh to point to their dataset location.
- Users need to set NUGRAPH_DIR, NUGRAPH_DATA and NUGRAPH_LOG environment variables.
- Training uses GPU by default
