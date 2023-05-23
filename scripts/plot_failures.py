#!/usr/bin/env python

import sys
import os
import time
import argparse
import torch
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import nugraph as ng
import numpy as np
import pandas as pd
import h5py
import plotly.express as px

def infer(args):

    # multiprocessing fixes for CPU threads on Wilson
    # https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
    print(f'Current Sharing Strategy: {torch.multiprocessing.get_sharing_strategy()}')
    torch.multiprocessing.set_sharing_strategy('file_system')
    print(f'Updated Sharing Strategy: {torch.multiprocessing.get_sharing_strategy()}')

    # Load dataset
    nudata = ng.data.H5DataModule('/data/processed/cosmics_final',
                                  batch_size=args.batch_size,
                                  planes=['u','v','y'],
                                  classes=['HIP','MIP','shower','michel','diffuse'],
                                  seed=1)

    model = ng.models.NuGraph2.load_from_checkpoint(args.checkpoint,
                               in_features=4,
                               node_features=args.node_features,
                               edge_features=args.edge_features,
                               sp_features=args.sp_features,
                               planes=nudata.planes,
                               classes=nudata.classes,
                               num_iters=5,
                               lr=args.learning_rate,
                               momentum=args.momentum,
                               # alpha=nudata.weights,
                               gamma=args.gamma)
    model.freeze()

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=args.logdir,
                                          name=args.name)
    profiler = pl.profilers.PyTorchProfiler(filename=f'{logger.log_dir}/profile.txt') \
            if args.profile else None
    if args.devices is None:
        print('No devices specified – running inference on CPU')

    accelerator = 'cpu' if args.devices is None else 'gpu'
    trainer = pl.Trainer(accelerator=accelerator,
                         devices=args.devices,
                         max_epochs=args.max_epochs,
                         logger=logger,
                         profiler=profiler)
    start = time.time()

    if args.nevents>0: 
        nudata.test_dataset.files.indices = nudata.test_dataset.files.indices[:args.nevents]

    predictions = trainer.predict(model, nudata.test_dataloader())

    end = time.time()
    itime = end - start

    ngraphs = nudata.test_dataset.len()
    print(f'inference for {ngraphs} events is {itime} s (that\'s {itime/ngraphs} s/graph')
    return predictions

def to_dataframe(g, evtmask):
    planes_arr = [None] * 3
    particle_dtype = pd.CategoricalDtype(['not_cosmic','cosmic'], ordered = True)
    for i,plane in enumerate(['u','v','y']):
        pmask = evtmask[i].tolist()
        preds = g[plane]['z_c'][pmask].round()
        #print(preds)
        d = {'local_wire': g[plane]['x'][pmask][:,0], 'local_time': g[plane]['x'][pmask][:,1], 
             'is_cosmic_truth':g[plane]['y_c'][pmask], 'is_cosmic_preds':preds,
             'prediction_strength': g[plane]['z_c'][pmask],
             #'truth_strength': g[plane]['z_c'][pmask][np.arange(len(g[plane]['z_c'][pmask])), g[plane]['y_s'][pmask]]
             }
        planes_arr[i] = pd.DataFrame(data = d)
        #print(planes_arr[i].is_cosmic_truth)
        planes_arr[i].is_cosmic_truth = planes_arr[i].is_cosmic_truth.astype(int)
        planes_arr[i].is_cosmic_truth = pd.Categorical(planes_arr[i].is_cosmic_truth).from_codes(codes = planes_arr[i].is_cosmic_truth, dtype = particle_dtype)
        planes_arr[i].is_cosmic_preds = planes_arr[i].is_cosmic_preds.astype(int)
        #print(planes_arr[i].is_cosmic_preds)
        planes_arr[i].is_cosmic_preds = pd.Categorical(planes_arr[i].is_cosmic_preds).from_codes(codes = planes_arr[i].is_cosmic_preds, dtype = particle_dtype)
        planes_arr[i]['local_plane']  = i
    return pd.concat(planes_arr)

def plot_event(g, evtmask, write=False, cnt=0):
  df = to_dataframe(g, evtmask)

  # only plot events with more than 25% failures
  failure_rate = 1 - (len(df[df['is_cosmic_truth'] == df['is_cosmic_preds']]) / len(df.index))
  #print(failure_rate)
  if failure_rate < 0.25:
      return None

  color_dict = {"pion" : "yellow",
      "muon" : "green",
      "kaon" : "black",
      "hadron" : "blue",
      "shower" : "red",
      "michel" : "purple",
      "delta" : "pink",
      "diffuse" : "orange",
      "invisible" : "white",
      "HIP": "blue",
      "MIP": "green",}
     
  # add another color dict for cosmics
  cosmic_color_dict = {"cosmic" : "red", "not_cosmic" : "green"}

  fig1 = px.scatter(df, x="local_wire", y="local_time", color="is_cosmic_truth", color_discrete_map=cosmic_color_dict, facet_col="local_plane", 
                 labels={
                     "local_wire": "Wire",
                     "local_time": "Time Tick",
                     "is_cosmic_truth": "Cosmic/Non-cosmic"
                 },
                 title="Ground Truth")

  fig2 = px.scatter(df, x="local_wire", y="local_time", color="is_cosmic_preds", color_discrete_map=cosmic_color_dict, facet_col="local_plane", hover_data=["is_cosmic_truth"],
                 labels={
                     "local_wire": "Wire",
                     "local_time": "Time Tick",
                     "is_cosmic_preds": "Cosmic/Non-cosmic"
                 },
                 title="Model Prediction")

  fig3 = px.scatter(df, x="local_wire", y="local_time", color="prediction_strength", color_continuous_scale=px.colors.diverging.RdYlGn_r, facet_col="local_plane", hover_data=["is_cosmic_preds","is_cosmic_truth"],
                 labels={
                     "local_wire": "Wire",
                     "local_time": "Time Tick",
                 },
                 title="Model Prediction Strength")

#  fig4 = px.scatter(df, x="local_wire", y="local_time", color="truth_strength", color_continuous_scale=px.colors.sequential.Greens, facet_col="local_plane", hover_data=["is_cosmic_preds","is_cosmic_truth", "prediction_strength"],
 #                labels={
  #                   "local_wire": "Wire",
   #                  "local_time": "Time Tick",
    #             },
     #            title="Model Truth Strength")


  for f in [fig1, fig2, fig3]:
    f.update_layout(
        width = 1200,
        height = 500,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="LightSteelBlue",
    )
    for a in f.layout.annotations:
      if a.text == "local_plane=0":
        a.text = "U Plane"
      elif a.text == "local_plane=1":
        a.text = "V Plane"
      else:
        a.text = "Y Plane"

  if write:
    #fig1.write_html(f'cnt-{cnt}_truth.html')
    #fig2.write_html(f'cnt-{cnt}_pred.html')
    #fig3.write_html(f'cnt-{cnt}_pred_strength.html')
    #fig4.write_html(f'cnt-{cnt}_truth_strength.html')
    fig1.write_image(f'cnt-{cnt}_truth.png')
    fig2.write_image(f'cnt-{cnt}_pred.png')
    fig3.write_image(f'cnt-{cnt}_pred_strength.png')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--name', type=str, required=True,
                        help='Training instance name, for logging purposes')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file for trained model')
    parser.add_argument('--devices', nargs='+', type=int, default=None,
                        help='List of devices to train with')
    parser.add_argument('--logdir', type=str, default='/nugraph/logs',
                        help='Output directory to write logs to')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='Enable PyTorch profiling')
    parser.add_argument('--nevents', type=int, default=10,
                        help='Limit number of events to process. If <0 all events are processed.')

    parser = ng.data.H5DataModule.add_data_args(parser)
    parser = ng.models.NuGraph2.add_model_args(parser)
    parser = ng.models.NuGraph2.add_train_args(parser)
    args = parser.parse_args()

    predictions = infer(args)

    evtcnt = 0
    for predbatch in predictions:
        for evt in range(len(predbatch)):
            evtmask = [ predbatch[p]['batch'] == evt for p in ['u','v','y'] ]

            plot_event(predbatch, evtmask, write=True, cnt=evtcnt)

            evtcnt += 1

            if args.nevents>0 and evtcnt > args.nevents: break

    print("Done!")
