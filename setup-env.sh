# setup script for nugraph on DSI cluster

export FERMI_GNN=/net/projects/fermi-gnn
export CONDA_PKGS_DIRS=$FERMI_GNN/conda/.pkgs
export NUGRAPH_DIR=$HOME/nugraph
export PYTHONPATH=$NUGRAPH_DIR/nugraph:$NUGRAPH_DIR/pynuml:$PYTHONPATH
export NUGRAPH_LOG=$FERMI_GNN/logs/kartavya

conda config --set env_prompt "({name}) "
conda activate $FERMI_GNN/conda/nugraph-25-summer