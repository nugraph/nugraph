
Reference to last quarter's README.md: [link](https://github.com/exatrkx/NuGraph/blob/vertex-param-search/README.md)

### Environment Setup

Make sure to be at your home directory on the cluster (e.g. /home/username).

1. Installing dependencies with Anaconda Client

    numl-dsi environment contains pytorch packages dependenceis for nugraph library.
    ```
    conda install -y anaconda-client
    conda env create numl/numl-dsi
    ```
2. Installing NuGraph in editable mode

    this allows you to access Fermilab researchers' latest update to nugraph library.
    ```
    git clone git@github.com:exatrkx/NuGraph.git
    conda activate numl-dsi
    pip install --no-deps -e ./NuGraph
    ```

### Data Source
We will be using the MicroBoone dataset. This [link](https://microboone.fnal.gov/documents-publications/public-datasets/) contains more information on the data. The data file is located at /net/projects/fermi-gnn/CHEP2023.gnn.h5 -- make sure you have access to the data (you can check by running touch /net/projects/fermi-gnn/CHEP2023.gnn.h5).

### Train models in interactive session
Some lessons we learned: requesting 60G memory is enough, and GPU is necessary.
1. Request a compute node

```
srun --pty \
    --time=7:00:00 \
    --partition=general \
    --nodes=1 \
    --gres=gpu:1 \
    --cpus-per-gpu=8 \
    --mem-per-cpu=60G 
    bash
```

2. Run training script
```
scripts/train.py \
    --logdir <path-to-logs-folder> \
    --name dsi-example \
    --version semantic-filter-vertex \
    --data-path /net/projects/fermi-gnn/CHEP2023.gnn.h5 \
    --semantic \
    --filter \
    --vertex
```

### Train models with scripts
Edit train_batch_dsi.sh accordingly, then:
#### LSTM aggregator
Submit the following command in terminal. Edit --version and --vertex-mlp-feats as necessary.
```
sbatch scripts/train_batch_dsi.sh --version lstm-32-mlp-64-sementic-filter --vertex-mlp-feats 64 --vertex-aggr lstm --vertex-lstm-feats 32 --semantic --filter
```
#### mean aggregator
Submit the following command in terminal. Edit --version and --vertex-mlp-feats as necessary.
```
sbatch scripts/train_batch_dsi.sh --version mean-mlp-64-sementic-filter --vertex-mlp-feats 64 --vertex-aggr mean --semantic --filter
```
#### attentional aggregator
Submit the following command in terminal. Edit --version and --vertex-mlp-feats as necessary.
```
sbatch scripts/train_batch_dsi.sh --version attentional-mlp-64-sementic-filter --vertex-mlp-feats 64 --vertex-aggr attn --semantic --filter
```
#### set transformer aggregator

### Compare models with Tensorboard
1. Insert random numbers for XXXX which you think no one else will use, and run from terminal (numl-dsi should be activated):
tensorboard --port XXXX --bind_all --logdir (Fill in your log directory address) --samples_per_plugin images=200

2. Copy and paste the created link into a web browser
