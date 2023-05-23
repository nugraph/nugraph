# DUNE Multihead Attention Message-Passing GNN

This branch contains the code to run the Liquid Argon TPC graph neural network. A Docker container containing the environment used to run this code can be found at `jhewes/pytorch-neutrinoml:1.9` on Docker hub.

This project is designed to classify and reconstruct neutrino interactions in a LArTPC. Each graph consists of a 2D representation of a neutrino interaction, where each node is a detector hit. Each edge connecting a pair of hits is labelled in truth according to particle type, and a multihead attention message-passing network is trained to classify graph edges.

## Opening a Docker container

After pulling down the Docker image from docker hub, one should be able to start the container using the script `run_docker_pytorch.sh`, providing an integer argument as an argument. This will start a container `<username>-gcn-<i>`, where `i` is the integer provided. This integer argument allows you to have multiple containers running simultaneously, if you so wish - start with `1`, and increment as necessary. `docker ps` will list all running containers.

Inside this script is the command that launches the container, and you should pay specific attention to the `-v /raid:/data` part. This argument mounts the directory `/raid` inside the container as `/data` – if working on a machine other than Heimdall, this argument should be changed to point to the directory containing your data, and the corresponding change should be made to `config/hit2d.yaml` in the `path` parameter under the `data` header on line 3. For instance, if your inputs were located in `/home/Desktop/data/processed/*.pt`, then you could change `run_docker_pytorch.sh` to mount the data directory as `-v /home/Desktop/data:/data` inside your container, and then set `path: /data/processed` in your config file, as `/data` will point towards `/home/Desktop/data` inside your container.

## Producing training files

Running `scripts/process.py` will take the raw HDF5 files containing information on simulated neutrino events, and preprocess them into the `.pt` files that are used for training. The user should edit this script to ensure the input and output paths for HDF5 and `.pt` files are correct. The HDF5 files for preprocessing are not publicly available, but if you need access please contact me (jhewes15@fnal.gov).

## Training the model

Running `scripts/train.py` will take the preprocessed files and train a model with them. Configuration options for training are available in `config/hit2d.yaml`; by default the code assumes you are working on a machine with a 16GB GPU.

There are several new flags added by the implementation of the 3D Network. 
- use\_spacepoints: \[True, False\] (turns on/off 3D SpacePoint Network)
- node\_conv: \[True, False\] (turns on/off inference in linear layers in Node Network)
The implementation of node\_conv is a bit kludgey as it should ONLY be used when 'use\_spacepoints' is True. Consider adding an error case here as the current code does not catch this.
- sp\_conv\_last: \[True, False\] (if True, 3D Network will only be used on the very last message passing iteration)
- pooling: \[linear, mean, max\] (type of pooling on downward pass)
- sp\_attention: \[True, False\] (turns on/off attention on spacepoints in downward pass)
- concat\_2d: \[True, False\] (turns on/off allowing SpacePoint Network to infer 3D Features from BOTH 2D and inferred 3D features)
- sp\_how: \[standard, flatten, classwise\] (functionally same as 'how' flag for 2D Network)

## Inference

Running `python scripts/plot.py` will run over the number of inputs specified in `max_inputs` under the `inference` header in `config/hit2d.yaml`. For each graph, it will draw the true edges, and also the graph edges as classified by the model. Likewise, `python scripts/confusion.py` will run over the entire validation dataset, and then write the results as a confusion matrix between true and predicted classes. In both cases, the output will be written to the `plots` directory.

