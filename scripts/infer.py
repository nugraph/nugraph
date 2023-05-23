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
import pynuml
import awkward as ak

Data = ng.data.H5DataModule
Model = ng.models.NuGraph2

def configure():
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file for trained model')
    parser = Data.add_data_args(parser)
    parser = Model.add_model_args(parser)
    return parser.parse_args()

def infer(args):

    planes=['u','v','y']
    classes=['HIP','MIP','shower','michel','diffuse']
    label_map = torch.tensor([1,1,0,0,2,3,1,4]).long()

    model = Model.load_from_checkpoint(
            args.checkpoint,
            in_features=4,
            node_features=args.node_feats,
            edge_features=args.edge_feats,
            sp_features=args.sp_feats,
            planes=planes,
            classes=classes,
            num_iters=5)
    model.freeze()

    testfiles = ['/data/withendpoints/slice/nue.h5','/data/withendpoints/slice/numu.h5']
    truthfilter = ng.transforms.TruthFilter(planes=planes)


    events = ak.ArrayBuilder()

    for fname in testfiles:
        print('process file: ', fname)

        events = ak.ArrayBuilder()

        f = pynuml.io.File(fname)
        producer = pynuml.process.HitGraphProducer(
                file=f,
                labeller=pynuml.labels.standard)
        for evt in f:
            if not evt: continue
            name, data = producer(evt)
            if not data: continue
            data = truthfilter(data) # strip cosmic information
            data = model(data)

            hit_id = torch.cat([data[p].id for p in planes])
            score = torch.cat([data[p].z_s.detach() for p in planes])
            df = pd.DataFrame({
                'hit_id': hit_id,
                'semantic_label': score.argmax(dim=1) })
            df['run'] = data['metadata'].run
            df['subrun'] = data['metadata'].subrun
            df['event'] = data['metadata'].event

            events.append(df.to_numpy())

        array = ak.flatten(events.snapshot(), axis=1)

        hf = h5py.File(fname.replace('.h5','_infer.h5'), 'w')
        hf.create_dataset('hit_id', data=array[:,0])
        hf.create_dataset('sem_pred', data=array[:,1])
        hf.create_dataset('run', data=array[:,2])
        hf.create_dataset('subrun', data=array[:,3])
        hf.create_dataset('event', data=array[:,4])
        hf.close()

if __name__ == '__main__':

    args = configure()
    infer(args)
