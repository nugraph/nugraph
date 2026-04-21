# Training and inference

Training the nugraph model requires a graph HDF5 file containing a processed dataset of event graphs. If you do not have access to such a file, and would like to create one, then please refer to the previous documentation on [graph processing](graph-processing).

## Model training

A simple training script is provided at `scripts/train.py` [here](https://github.com/nugraph/nugraph/blob/main/scripts/train.py). A simple usage example is

```bash
scripts/train.py --device 0 --data-path /path/to/dataset.gnn.h5 --semantic
```

The `--device` option specifies the integer index of the GPU device to use to train. If this argument is omitted, model training will be performed on CPU instead.

```{admonition} Note on GPU devices
:class: warning

If you find that model training is proceeding very slowly, double-check that you included the `--device` argument. Training a model without any GPU devices will cause training to be prohibitively slow, and is only provided for debugging purposes.
```

Decoders to include in model training must be specified as arguments to the script – for instance, `--semantic --filter` will train the semantic and filter decoders, while `--semantic --instance --vertex` will train the semantic, instance and vertex decoders. Failing to pass any decoders as arguments will cause the training script to print an error message and then exit.

A full list of available arguments can be accessed simply by running

```bash
scripts/train.py --help
```

### Metric logging

The NuGraph3 architecture uses [Weights and Biases](https://wandb.ai) to log training metrics. The first time you start training, Weights and Biases will prompt you to register your account on the command line. After logging in, the metrics from your training runs will be logged to the browser interface on their website.

If you do not wish metrics to be logged, or in the case of issues connecting to Weights and Biases, you may disable metric logging by passing the `--offline` option to the training script.

## Inference on a trained model

During training, model parameter values are written to a checkpoint file with a `.ckpt` extension. Once training is complete, this checkpoint file can be used to test the performance of the trained model. A simple script exists to run inference over the test dataset using the trained model parameters, which can be found at `scripts/test.py` [here](https://github.com/nugraph/nugraph/blob/main/scripts/test.py).

This script can be run using

```bash
scripts/test.py --device <N> --checkpoint /path/to/checkpoint.ckpt --outfile <output.h5>
```

where `<N>` is the index of the GPU device to run inference with, `/path/to/checkpoint.ckpt` is the path to the checkpoint file containing the trained model, and `<output.h5> is the output HDF5 file to which model predictions will be written.

One can also run inference interactively using the ploting notebook `notebooks/test.ipynb` [here](https://github.com/nugraph/nugraph/blob/main/notebooks/plot.ipynb). This notebook will iterate interactively over events in the training dataset, passing them through the model to run inference, and then visualizing them using plotting tools. The user can use this notebook to visualize model predictions and compare them to truth labels.
