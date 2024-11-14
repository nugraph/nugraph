from typing import Any, Callable
import numpy as np
import pandas as pd

import torch
import torch_geometric as pyg

from .base import ProcessorBase

class HitGraphProducer(ProcessorBase):
    '''Process event into graphs'''

    def __init__(self,
                 file: 'pynuml.io.File',
                 semantic_labeller: Callable = None,
                 event_labeller: Callable = None,
                 label_vertex: bool = False,
                 label_position: bool = False,
                 planes: list[str] = ['u','v','y'],
                 node_pos: list[str] = ['local_wire','local_time'],
                 pos_norm: list[float] = [0.3,0.055],
                 node_feats: list[str] = ['integral','rms'],
                 lower_bound: int = 20,
                 store_detailed_truth: bool = False):

        self.semantic_labeller = semantic_labeller
        self.event_labeller = event_labeller
        self.label_vertex = label_vertex
        self.label_position = label_position
        self.planes = planes
        self.node_pos = node_pos
        self.pos_norm = torch.tensor(pos_norm).float()
        self.node_feats = node_feats
        self.lower_bound = lower_bound
        self.store_detailed_truth = store_detailed_truth

        self.transform = pyg.transforms.Compose((
            pyg.transforms.Delaunay(),
            pyg.transforms.FaceToEdge()))

        super().__init__(file)

    @property
    def columns(self) -> dict[str, list[str]]:
        groups = {
            'hit_table': ['hit_id','local_plane','local_time','local_wire','integral','rms'],
            'spacepoint_table': []
        }
        if self.semantic_labeller:
            groups['particle_table'] = ['g4_id','parent_id','type','momentum','start_process','end_process']
            groups['edep_table'] = []
        if self.event_labeller:
            groups['event_table'] = ['is_cc', 'nu_pdg']
        if self.label_vertex:
            keys = ['nu_vtx_corr','nu_vtx_wire_pos','nu_vtx_wire_time']
            if 'event_table' in groups:
                groups['event_table'].extend(keys)
            else:
                groups['event_table'] = keys
        if self.label_position:
            groups["edep_table"] = []
        return groups

    @property
    def metadata(self):
        metadata = { 'planes': self.planes }
        if self.semantic_labeller is not None:
            metadata['semantic_classes'] = self.semantic_labeller.labels[:-1]
        if self.event_labeller is not None:
            metadata['event_classes'] = self.event_labeller.labels
        return metadata

    def __call__(self, evt: 'pynuml.io.Event') -> tuple[str, Any]:

        if self.event_labeller or self.label_vertex:
            event = evt['event_table'].squeeze()

        hits = evt['hit_table']
        spacepoints = evt['spacepoint_table'].reset_index(drop=True)

        # discard any events with pathologically large hit integrals
        # this is a hotfix that should be removed once the dataset is fixed
        if hits.integral.max() > 1e6:
            print('found event with pathologically large hit integral, skipping')
            return evt.name, None

        # handle energy depositions
        if self.semantic_labeller:
            edeps = evt['edep_table']
            energy_col = 'energy' if 'energy' in edeps.columns else 'energy_fraction' # for backwards compatibility
        
            # get ID of max particle
            g4_id = edeps[[energy_col, 'g4_id', 'hit_id']]
            g4_id = g4_id.sort_values(by=[energy_col],
                                      ascending=False,
                                      kind='mergesort').drop_duplicates('hit_id')
            hits = g4_id.merge(hits, on='hit_id', how='right')
            
            # charge-weighted average of 3D position
            if self.label_position:
                edeps = edeps[["hit_id", "energy", "x_position", "y_position", "z_position"]]
                for col in ["x_position", "y_position", "z_position"]:
                    edeps.loc[:, col] *= edeps.energy
                edeps = edeps.groupby("hit_id").sum()
                for col in ["x_position", "y_position", "z_position"]:
                    edeps.loc[:, col] /= edeps.energy
                edeps = edeps.drop("energy", axis="columns")
                hits = edeps.merge(hits, on="hit_id", how="right")

            hits['filter_label'] = ~hits[energy_col].isnull()
            hits = hits.drop(energy_col, axis='columns')

        # reset spacepoint index
        spacepoints = spacepoints.reset_index(names='index_3d')

        # skip events with fewer than lower_bnd simulated hits in any plane.
        # note that we can't just do a pandas groupby here, because that will
        # skip over any planes with zero hits
        for i in range(len(self.planes)):
            planehits = hits[hits.local_plane==i]
            nhits = planehits.filter_label.sum() if self.semantic_labeller else planehits.shape[0]
            if nhits < self.lower_bound:
                return evt.name, None

        # get labels for each particle
        if self.semantic_labeller:
            particles = self.semantic_labeller(evt['particle_table'])
            try:
                hits = hits.merge(particles, on='g4_id', how='left')
            except:
                print('exception occurred when merging hits and particles')
                print('hit table:', hits)
                print('particle table:', particles)
                print('skipping this event')
                return evt.name, None
            mask = (~hits.g4_id.isnull()) & (hits.semantic_label.isnull())
            if mask.any():
                print(f'found {mask.sum()} orphaned hits.')
                return evt.name, None
            del mask

        data = pyg.data.HeteroData()

        # event metadata
        r, sr, e = evt.event_id
        data['metadata'].run = r
        data['metadata'].subrun = sr
        data['metadata'].event = e

        # spacepoint nodes
        if "position_x" in spacepoints.keys():
            data["sp"].pos = torch.tensor(spacepoints[[f"position_{c}" for c in ("x", "y", "z")]].values).float()
        else:
            data['sp'].num_nodes = spacepoints.shape[0]

        hits = hits.reset_index(names="index_2d")

        # node position
        hits[self.node_pos] *= self.pos_norm
        data["hit"].pos = torch.tensor(hits[self.node_pos].values).float()

        # plane indices
        data["hit"].plane = torch.tensor(hits["local_plane"].values, dtype=torch.long)

        # node features
        data["hit"].x = torch.tensor(hits[self.node_feats].values).float()

        # node true position
        if self.label_position:
            data["hit"].c = torch.tensor(hits[["x_position", "y_position", "z_position"]].values).float()

        # hit indices
        data["hit"].id = torch.tensor(hits['hit_id'].values).long()

        # 2D graph edges
        data["hit", "delaunay", "hit"].edge_index = self.transform(data["hit"]).edge_index
        edge_plane = []
        for i, plane_hits in hits.groupby("local_plane"):
            tmp = pyg.data.Data()
            tmp.index_2d = torch.tensor(plane_hits.index_2d.values).long()
            tmp.pos = torch.tensor(plane_hits[self.node_pos].values).float()
            edge_plane.append(tmp.index_2d[self.transform(tmp).edge_index])
        data["hit", "delaunay-planar", "hit"].edge_index = torch.cat(edge_plane, dim=1)

        # 3D graph edges
        edge_nexus = []
        for i, plane_hits in hits.groupby("local_plane"):
            p = self.planes[i]
            edge = spacepoints.merge(hits[['hit_id','index_2d']].add_suffix(f'_{p}'),
                                     on=f'hit_id_{p}',
                                     how='inner')
            edge = edge[[f'index_2d_{p}','index_3d']].values.transpose()
            edge = torch.tensor(edge) if edge.size else torch.empty((2,0))
            edge_nexus.append(edge.long())
        data["hit", "nexus", "sp"].edge_index = torch.cat(edge_nexus, dim=1)

        # add edges to event node
        data["evt"].num_nodes = 1
        lo = torch.arange(data["hit"].num_nodes, dtype=torch.long)
        hi = torch.zeros(data["hit"].num_nodes, dtype=torch.long)
        data["hit", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)
        lo = torch.arange(data["sp"].num_nodes, dtype=torch.long)
        hi = torch.zeros(data["sp"].num_nodes, dtype=torch.long)
        data["sp", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)

        # truth information
        if self.semantic_labeller:
            data["hit"].y_semantic = torch.tensor(hits['semantic_label'].fillna(-1).values).long()
            data["hit"].y_instance = torch.tensor(hits['instance_label'].fillna(-1).values).long()
            if self.store_detailed_truth:
                data["hit"].g4_id = torch.tensor(hits['g4_id'].fillna(-1).values).long()
                data["hit"].parent_id = torch.tensor(hits['parent_id'].fillna(-1).values).long()
                data["hit"].pdg = torch.tensor(hits['type'].fillna(-1).values).long()

        # event label
        if self.event_labeller:
            data['evt'].y = torch.tensor(self.event_labeller(event)).long().reshape([1])

        # 3D vertex truth
        if self.label_vertex:
            vtx_3d = [ [ event.nu_vtx_corr_x, event.nu_vtx_corr_y, event.nu_vtx_corr_z ] ]
            data['evt'].y_vtx = torch.tensor(vtx_3d).float()

        return evt.name, data
