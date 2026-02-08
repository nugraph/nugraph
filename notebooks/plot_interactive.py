##### Instructions #####
# First request interactive resources from slurm using the terminal:
# example: srun --partition=general --time=03:00:00 --gres=gpu:1 --nodes=1 --cpus-per-task=16 --ntasks=1 --mem=60g --pty /bin/bash
# Once resources are granted, run the following lines to set up conda environment:
# source /etc/profile.d/conda.sh
# conda activate /net/projects2/fermi2526/conda/nugraph-25-10
# Ensure that dash is installed (pip install dash)
# Set settings below (data path, model checkpoint path, path to nugraph, path to pynuml)
# Run this file: python /path/to/plot_interactive.py
# Wait until you see 'Dash is running on http://0.0.0.0:8050/'
# In a local terminal, run: ssh -N -L [local-port]:[compute-node]:8050 [user]@fe01.ds.uchicago.edu
# Navigate to http://localhost:8050 in your browser

##### Settings #####
data_path = "/net/projects2/fermi2526/data/uboone-opendata-19be46d8.gnn.h5"
checkpoint_path = "/net/projects2/fermi2526/logs/aidanjl1/first_test_batch_V10/checkpoints/epoch=24-step=155200.ckpt"
nugraph_path = "/home/mlalwani/nugraph_dsifall/nugraph"
pynuml_path = "/home/mlalwani/nugraph_dsifall/pynuml"

##### Start Message #####
print("Program is running..")

##### Imports #####
import os
import sys
try:
    import dash
except ImportError:
    print("Dash is not installed. Install using 'pip install dash' to continue.")
    sys.exit()
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
sys.path.append(nugraph_path)
sys.path.append(pynuml_path)
print("Importing nugraph. This may take up to 10 minutes, especially the first time...")
import nugraph as ng
print("Nugraph imported.")
import pynuml
import torch
print("Imports done.", flush=True)

##### Loading Data and Model #####
print("Loading data and model...")
Data = ng.data.NuGraphDataModule
Model = ng.models.NuGraph3
nudata = Data(
    model=Model,
    data_path=data_path,
)
ckpt = os.path.expandvars(checkpoint_path)
model = Model.load_from_checkpoint(ckpt, map_location="cpu")
model.freeze()

##### Plotting #####
print("Loading plotting function...")

plotter = pynuml.plot.GraphPlot(
    planes=nudata.planes,
    classes=nudata.semantic_classes,
)

dataset = nudata.test_dataset
idx = 0

def make_figure(i, target, how, filter):
    data = dataset[i]
    model(data)
    figw = plotter.plot(
        data,
        target=target,
        how=how,
        filter=filter,
    )
    return go.Figure(figw)

##### Setting up Dash #####
print("Setting up dash...", flush=True)
print("Max event # = 200. Increase in plot_dash.py file if needed.")
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("Event Displays - NuGraph3"),
    dcc.Slider(
        id="event",
        min=0,
        max=200,
        step=1,
        value=0,
    ),
    dcc.Dropdown(
        id="target",
        options=[
            {"label": "Semantic", "value": "semantic"},
            {"label": "Instance", "value": "instance"},
            {"label": "Hits", "value": "hits"},
            {"label": "Filter", "value": "filter"},
        ],
        value="semantic",
    ),
    dcc.Dropdown(
        id="how",
        options=[
            {"label": "True (semantic, instance, filter)", "value": "true"},
            {"label": "Predicted (semantic, instance)", "value": "pred"},
            {"label": "Beta (instance)", "value": "beta"},
            {"label": "PCA (instance)", "value": "pca"},
        ],
        value="true",
    ),
    dcc.Dropdown(
        id="filter",
        options=[
            {"label": "None (always works)", "value": "none"},
            {"label": "Show", "value": "show"},
            {"label": "True", "value": "true"},
            {"label": "Pred", "value": "pred"},
        ],
        value="none",
    ),
    dcc.Graph(id="graph"),
])

@app.callback(
    Output("graph", "figure"),
    Input("event", "value"),
    Input("target", "value"),
    Input("how", "value"),
    Input("filter", "value"),
)
def update_graph(i, target_value, how_value, filter_value):
    return make_figure(i, target_value, how_value, filter_value)

if __name__ == "__main__":
    print("Running dash server. Tunnel to compute node using ssh -N -L [local-port]:[compute-node]:8050 [user]@fe01.ds.uchicago.edu. Open http://localhost:[local-port] to see plots.", flush=True)
    app.run(host="0.0.0.0", port=8050, debug=False)
