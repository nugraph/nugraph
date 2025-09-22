from typing import Any, Callable

import torch
import torch_geometric as pyg
import pandas as pd

from ..data import NuGraphData
from .base import ProcessorBase

class HitGraphProducer(ProcessorBase):
    '''Process event into graphs'''

    def __init__(self,
                 file: 'pynuml.io.File',
                 semantic_labeller: Callable = None,
                 event_labeller: Callable = None,
                 label_vertex: bool = False,
                 label_position: bool = False,
                 optical: bool = False,
                 planes: list[str] = ['u','v','y'],
                 node_feats: list[str] = ['integral','rms','tpc'],
                 lower_bound: int = 20,
                 store_detailed_truth: bool = False):

        self.semantic_labeller = semantic_labeller
        self.event_labeller = event_labeller
        self.label_vertex = label_vertex
        self.label_position = label_position
        self.optical = optical
        self.planes = planes
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
            'hit_table': [],
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
        if self.optical:
            groups["ophit_table"] = []
            groups["opflash_table"] = []
            groups["opflashsumpe_table"] = []
        return groups

    @property
    def metadata(self):
        metadata = dict(planes=self.planes, gen=torch.tensor([2]))
        if self.semantic_labeller is not None:
            metadata['semantic_classes'] = self.semantic_labeller.labels[:-1]
        if self.event_labeller is not None:
            metadata['event_classes'] = self.event_labeller.labels
        return metadata

    def __call__(self, evt: 'pynuml.io.Event') -> tuple[str, Any]:

        if self.event_labeller or self.label_vertex:
            event = evt['event_table'].squeeze()

        # support different generations of event HDF5 format
        hits = evt['hit_table']
        if "global_plane" in hits.columns:
            plane_key, proj_key, drift_key = "global_plane", "global_wire", "global_time"
        elif "local_plane" in hits.columns:
            plane_key, proj_key, drift_key = "local_plane", "local_wire", "local_time"
        else:
            plane_key, proj_key, drift_key = "view", "proj", "drift"

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
            planehits = hits[hits[plane_key]==i]
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

        data = NuGraphData()

        # event metadata
        r, sr, e = evt.event_id
        data['metadata'].run = r
        data['metadata'].subrun = sr
        data['metadata'].event = e

        # spacepoint nodes
        if "position_x" in spacepoints.keys() and len(spacepoints)>0:
            data["sp"].pos = torch.tensor(spacepoints[[f"position_{c}" for c in ("x", "y", "z")]].values).float()
        else:
            data['sp'].num_nodes = spacepoints.shape[0]

        hits = hits.reset_index(names="index_2d")

        node_pos = [proj_key, drift_key]

        # node position
        data["hit"].plane = torch.tensor(hits[plane_key].values, dtype=torch.long)
        data["hit"].pos = torch.tensor(hits[node_pos].values, dtype=torch.float)

        # node features
        node_feats = self.node_feats + [plane_key, proj_key, drift_key]
        data["hit"].x = torch.tensor(hits[node_feats].values).float()

        # node true position
        if self.label_position:
            data["hit"].y_position = torch.tensor(hits[["x_position", "y_position", "z_position"]].values).float()

        # hit indices
        data["hit"].id = torch.tensor(hits['hit_id'].values).long()

        # 2D graph edges
        data["hit", "delaunay", "hit"].edge_index = self.transform(data["hit"]).edge_index
        edge_plane = []
        for i, view_hits in hits.groupby(plane_key):
            tmp = pyg.data.Data()
            tmp.index_2d = torch.tensor(view_hits.index_2d.values).long()
            tmp.pos = torch.tensor(view_hits[node_pos].values).float()
            edge_plane.append(tmp.index_2d[self.transform(tmp).edge_index])
        data["hit", "delaunay-planar", "hit"].edge_index = torch.cat(edge_plane, dim=1)

        # 3D graph edges
        edge_nexus = []
        for i, view_hits in hits.groupby(plane_key):
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
            y = torch.tensor(hits['instance_label'].fillna(-1).values).long()
            mask = y != -1
            y = y[mask]
            instances = y.unique()
            # remap instances
            imax = instances.max() + 1 if instances.size(0) else 0
            if instances.size(0) != imax:
                remap = torch.full((imax,), -1, dtype=torch.long)
                remap[instances] = torch.arange(instances.size(0))
                y = remap[y]
            data["particle-truth"].num_nodes = instances.size(0)
            edges = torch.stack((mask.nonzero().squeeze(1), y), dim=0).long()
            data["hit", "cluster-truth", "particle-truth"].edge_index = edges
            if self.store_detailed_truth:
                data["hit"].g4_id = torch.tensor(hits['g4_id'].fillna(-1).values).long()
                data["hit"].parent_id = torch.tensor(hits['parent_id'].fillna(-1).values).long()
                data["hit"].pdg = torch.tensor(hits['type'].fillna(-1).values).long()

        # optical system
        if self.optical:

            ophits = evt["ophit_table"]
            sum_pe = evt["opflashsumpe_table"]
            opflash = evt["opflash_table"]

            # skip events with no flash
            if opflash.shape[0]==0:
                return evt.name, None

            # node position
            data["ophit"].pos = torch.tensor(ophits[["wire_pos_0", "wire_pos_1", "wire_pos_2"]].values).float()
            data["flash"].pos = torch.tensor(opflash[["wire_pos_0", "wire_pos_1", "wire_pos_2"]].values).float()

            if "pos_y" in sum_pe.columns:
                data["pmt"].pos = torch.tensor(sum_pe[["pos_y", "pos_z"]].values).float()
            else:
                #hardcoded positions for MicroBooNE's opdets
                opdet_pos_y = torch.tensor([55.267144, 55.962509, 27.555318, -0.850317, -28.561692, -56.620694, -56.447756, 55.442895, 55.789304, -0.675445,
                                            0.017374, -56.275066, -56.274171, 55.616099, 55.616099, -0.50224, -1.021855, -56.100966, -56.100966, 54.750076,
                                            54.749983, -0.675445, -0.84865, -56.96699, -56.274171, 55.096391, 55.269595, 27.556793, -0.502415, -28.734833,
                                            -56.274171, -56.620838])
                opdet_pos_z = torch.tensor([951.85, 911.05, 989.65, 865.45, 990.25, 951.85, 911.95, 751.75, 710.95, 796.15,
                                            664.15, 752.05, 711.25, 540.85, 500.05, 585.25, 452.95, 540.55, 500.35, 328.15,
                                            287.95, 373.75, 242.05, 328.45, 287.65, 128.35,  87.85,  51.25, 173.65,  50.35,
                                            128.05,  87.85])
                data["pmt"].pos = torch.stack([opdet_pos_y[sum_pe["pmt_channel"].values], opdet_pos_z[sum_pe["pmt_channel"].values]], dim=1)

            # optical node features (not including the positions)
            data["ophit"].x = torch.cat([data["ophit"].pos,
            torch.tensor(ophits[["amplitude", "area",  "pe", "peaktime", "width"]].values).float()],dim=1)
            data["flash"].x = torch.cat([data["flash"].pos,torch.tensor(opflash[["time", "time_width", "totalpe", "y_center", "y_width", "z_center", "z_width"]].values).float()],dim=1)
            data["pmt"].x = torch.cat([data["pmt"].pos,torch.tensor(sum_pe[["pmt_channel", "sumpe"]].values).float()],dim=1)

            # ophit to pmt edges
            edge1 = torch.tensor(ophits[["hit_id","sumpe_id"]].values.transpose())
            mask = edge1[1,:]>=0
            mask = torch.nonzero(mask)
            edge1 = torch.squeeze(edge1[:,mask])
            data["ophit", "in", "pmt"].edge_index = edge1.long()

            # pmt to flash edges
            edge2 = torch.tensor(sum_pe[["sumpe_id", "flash_id"]].values.transpose())
            data["pmt", "in", "flash"].edge_index = edge2.long()

            # flash to event edges
            edge3 = torch.tensor([opflash["flash_id"].values[0], 0])
            data["flash", "in", "evt"].edge_index = edge3

            # nexus to pmt edges
            spacepoints_nodes = torch.tensor(spacepoints[["position_y", "position_z"]].values)
            distances = torch.cdist( spacepoints_nodes, data["pmt"].pos, p=2)
            nnear = 2
            _, nearest_indices = torch.topk(distances, nnear, largest=False, dim=1)

            spacepoints_indices = torch.arange( spacepoints_nodes.size(0) ).repeat_interleave(nnear)
            opflashsumpe_indices = nearest_indices.flatten()

            edges = torch.stack([spacepoints_indices, opflashsumpe_indices], dim=0)
            data["sp", "knn", "pmt"].edge_index = edges.long()

        # event label
        if self.event_labeller:
            # pylint: disable=possibly-used-before-assignment
            data['evt'].y = torch.tensor(self.event_labeller(event)).long().reshape([1])

        # 3D vertex truth
        if self.label_vertex:
            vtx_3d = [ [ event.nu_vtx_corr_x, event.nu_vtx_corr_y, event.nu_vtx_corr_z ] ]
            data['evt'].y_vtx = torch.tensor(vtx_3d).float()

        return evt.name, data
