
Reference to last quarter's README.md: [link](https://github.com/exatrkx/NuGraph/blob/vertex-param-search/README.md)

### Environment Setup

[TODO]insert instructions on how to prepare for installation such as: make sure to have Anaconda installed, make sure to have access to the data, be at your home directory on the cluster...

1. installing dependencies with Anaconda Client

    [TODO]insert background of the numl-dsi environment...
    ```
    conda install -y anaconda-client
    conda env create numl/numl-dsi
    ```
2. installing NuGraph in editable mode

    [TODO]insert explanation on why NuGraph needs to be installed again...
    ```
    git clone git@github.com:exatrkx/NuGraph.git
    conda activate numl-dsi
    pip install --no-deps -e ./NuGraph
    ```

### Data Source
[TODO]insert description of data

### Helpful External Links
[TODO]insert helpful links, maybe better to separate them in sections

### Train models in interactive session
[TODO]insert on lessons learned when using interactive session
1. Request a compute node

```
srun --pty \
    --time=7:00:00 \
    --partition=general \
    --nodes=1 \
    --gres=gpu:1 \
    --constraint=a100 \
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
[TODO]insert instructions and scripts under each section

***There should be description for every file created this quarter***
#### LSTM aggregator
#### mean aggregator
#### attentional aggregator
#### set transformer aggregator

### Compare models with Tensorboard
1. Into Terminal: 
```tensorboard --port XXXX --bind_all --logdir /net/projects/fermi-2/logs --samples_per_plugin ‘images=200’```
Insert random numbers for XXXX which you think no one else will use
2. Copy and paste the created link into a web browser

### Contacts
- name and email
- name and email
- name and email