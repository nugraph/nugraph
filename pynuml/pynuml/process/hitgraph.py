"""Class to produce particle hit graph objects"""

from typing import Any, Callable

import torch
import torch_geometric as pyg

from ..io import File, Event
from ..data import NuGraphData
from .base import ProcessorBase

POS3D = ["x_position", "y_position", "z_position"]

class HitGraphProducer(ProcessorBase):
    """
    Process event into graph object

    Args:
        file: input pynuml file
        semantic_labeller: semantic labeller instance
        event_labeller: event labeller instance
        label_vertex: whether to label true vertex position
        label_position: whether to label true hit position
        planes: list of detector planes
        node_feats: list of node feature names
        lower_bound: minimum hits required on each plane
        store_detailed_truth: whether to save detailed truth information
    """

    def __init__(self,
                 file: File,
                 semantic_labeller: Callable = None,
                 event_labeller: Callable = None,
                 label_vertex: bool = False,
                 label_position: bool = False,
                 planes: list[str] = ["u", "v", "y"],
                 node_feats: list[str] = ["integral", "rms", "tpc"],
                 lower_bound: int = 20,
                 store_detailed_truth: bool = False):

        self.semantic_labeller = semantic_labeller
        self.event_labeller = event_labeller
        self.label_vertex = label_vertex
        self.label_position = label_position
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
            "hit_table": [],
            "spacepoint_table": []
        }
        if self.semantic_labeller:
            groups["particle_table"] = ["g4_id", "parent_id", "type", "momentum",
                                        "start_process", "end_process"]
            groups["edep_table"] = []
        if self.event_labeller:
            groups["event_table"] = ["is_cc", "nu_pdg"]
        if self.label_vertex:
            keys = ["nu_vtx_corr", "nu_vtx_wire_pos", "nu_vtx_wire_time"]
            if "event_table" in groups:
                groups["event_table"].extend(keys)
            else:
                groups["event_table"] = keys
        if self.label_position:
            groups["edep_table"] = []
        return groups

    @property
    def metadata(self):
        metadata = {"planes": self.planes, "gen": torch.tensor([2])}
        if self.semantic_labeller is not None:
            metadata["semantic_classes"] = self.semantic_labeller.labels[:-1]
        if self.event_labeller is not None:
            metadata["event_classes"] = self.event_labeller.labels
        return metadata

    def __call__(self, evt: Event) -> tuple[str, Any]:

        # hit dataframe
        hits = evt["hit_table"]
        hits = hits.rename(columns={
            "local_plane": "view",
            "local_wire": "proj",
            "local_time": "drift",
        })

        # spacepoint dataframe
        spacepoints = evt["spacepoint_table"].reset_index(drop=True)
        spacepoints = spacepoints.rename(
            columns={f"position_{c}": f"{c}_position" for c in ("x", "y", "z")})

        # event dataframe
        if self.event_labeller or self.label_vertex:
            event = evt["event_table"].squeeze()

        # energy deposition dataframe
        if self.semantic_labeller:
            edeps = evt["edep_table"]
            edeps = edeps.rename(columns={"energy_fraction": "energy"})

        # true particle dataframe
        if self.semantic_labeller:
            particles = evt["particle_table"]

        # discard any events with pathologically large hit integrals
        # this is a hotfix that should be removed once the dataset is fixed
        if hits.integral.max() > 1e6:
            print("found event with pathologically large hit integral, skipping")
            return evt.name, None

        # handle energy depositions
        if self.semantic_labeller:

            # get ID of max particle
            g4_id = edeps[["energy", "g4_id", "hit_id"]]
            g4_id = g4_id.sort_values(by=["energy"],
                                      ascending=False,
                                      kind="mergesort").drop_duplicates("hit_id")
            hits = g4_id.merge(hits, on="hit_id", how="right")

            # charge-weighted average of 3D position
            if self.label_position:
                edeps = edeps[["hit_id", "energy"] + POS3D]
                for col in POS3D:
                    edeps.loc[:, col] *= edeps.energy
                edeps = edeps.groupby("hit_id").sum()
                for col in POS3D:
                    edeps.loc[:, col] /= edeps.energy
                edeps = edeps.drop("energy", axis="columns")
                hits = edeps.merge(hits, on="hit_id", how="right")

            hits["filter_label"] = ~hits["energy"].isnull()
            hits = hits.drop("energy", axis="columns")

        # reset spacepoint index
        spacepoints = spacepoints.reset_index(names="index_3d")

        # skip events with fewer than lower_bnd simulated hits in any plane.
        # note that we can't just do a pandas groupby here, because that will
        # skip over any planes with zero hits
        for i in range(len(self.planes)):
            planehits = hits[hits["view"]==i]
            nhits = planehits.filter_label.sum() if self.semantic_labeller else planehits.shape[0]
            if nhits < self.lower_bound:
                return evt.name, None

        # get labels for each particle
        if self.semantic_labeller:
            particles = self.semantic_labeller(particles)
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

        # create data object and declare shorthand for node stores
        data = NuGraphData()
        h = data["hit"]
        n = data["nexus"]
        e = data["evt"]
        p = data["particle"]

        # event metadata
        data.run, data.subrun, data.event = evt.event_id

        # nexus nodes
        n.num_nodes = torch.tensor(len(spacepoints.index), dtype=torch.long)
        if "x_position" in spacepoints.keys():
            n.y_position = torch.tensor(spacepoints[POS3D].values, dtype=torch.float)

        hits = hits.reset_index(names="index_2d")

        # node attributes
        h.num_nodes = torch.tensor(len(hits.index), dtype=torch.long)
        h.plane = torch.tensor(hits["view"].values, dtype=torch.long)
        h.proj = torch.tensor(hits["proj"].values, dtype=torch.float)
        h.drift = torch.tensor(hits["drift"].values, dtype=torch.float)
        h.integral = torch.tensor(hits["integral"].values, dtype=torch.float)
        h.rms = torch.tensor(hits["rms"].values, dtype=torch.float)
        h.tpc = torch.tensor(hits["tpc"].values, dtype=torch.long)

        # node true position
        if self.label_position:
            if "x_position" not in hits.keys():
                raise RuntimeError(("Position labelling was enabled, but position "
                                    "truth is not available in event HDF5 file."))
            h.y_position = torch.tensor(hits[POS3D].values, dtype=torch.float)

        # hit indices
        h.id = torch.tensor(hits["hit_id"].values).long()

        # 2D graph edges
        h.pos = torch.stack((h.proj, h.drift), dim=1)
        data["hit", "delaunay", "hit"].edge_index = self.transform(h).edge_index
        edge_plane = []
        for i, view_hits in hits.groupby("view"):
            tmp = pyg.data.Data()
            tmp.index_2d = torch.tensor(view_hits.index_2d.values, dtype=torch.long)
            tmp.pos = torch.tensor(view_hits[["proj", "drift"]].values, dtype=torch.float)
            edge_plane.append(tmp.index_2d[self.transform(tmp).edge_index])
        data["hit", "delaunay-planar", "hit"].edge_index = torch.cat(edge_plane, dim=1)
        del h.pos

        # 3D graph edges
        edge_nexus = []
        for i, view_hits in hits.groupby("view"):
            plane = self.planes[i]
            edge = spacepoints.merge(hits[["hit_id", "index_2d"]].add_suffix(f"_{plane}"),
                                     on=f"hit_id_{plane}",
                                     how="inner")
            edge = edge[[f"index_2d_{plane}","index_3d"]].values.transpose()
            edge = torch.tensor(edge) if edge.size else torch.empty((2,0))
            edge_nexus.append(edge.long())
        data["hit", "nexus", "nexus"].edge_index = torch.cat(edge_nexus, dim=1)

        # add edges to event node
        e.num_nodes = torch.tensor(1, dtype=torch.long)
        lo = torch.arange(h.num_nodes, dtype=torch.long)
        hi = torch.zeros(h.num_nodes, dtype=torch.long)
        data["hit", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)
        lo = torch.arange(n.num_nodes, dtype=torch.long)
        hi = torch.zeros(n.num_nodes, dtype=torch.long)
        data["nexus", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)

        # truth information
        if self.semantic_labeller:
            h.y_semantic = torch.tensor(hits['semantic_label'].fillna(-1).values, dtype=torch.long)
            y = torch.tensor(hits['instance_label'].fillna(-1).values, dtype=torch.long)
            mask = y != -1
            y = y[mask]
            instances = y.unique()
            # remap instances
            imax = instances.max() + 1 if instances.size(0) else 0
            if instances.size(0) != imax:
                remap = torch.full((imax,), -1, dtype=torch.long)
                remap[instances] = torch.arange(instances.size(0))
                y = remap[y]
            p.num_nodes = torch.tensor(instances.size(0), dtype=torch.long)
            edges = torch.stack((mask.nonzero().squeeze(1), y), dim=0).long()
            data["hit", "true-particle", "particle"].edge_index = edges
            if self.store_detailed_truth:
                h.g4_id = torch.tensor(hits['g4_id'].fillna(-1).values).long()
                h.parent_id = torch.tensor(hits['parent_id'].fillna(-1).values).long()
                h.pdg = torch.tensor(hits['type'].fillna(-1).values).long()

        # event label
        if self.event_labeller:
            # pylint: disable=possibly-used-before-assignment
            e.y = torch.tensor(self.event_labeller(event)).long().reshape([1])

        # 3D vertex truth
        if self.label_vertex:
            vtx_3d = [ [ event.nu_vtx_corr_x, event.nu_vtx_corr_y, event.nu_vtx_corr_z ] ]
            e.y_vtx = torch.tensor(vtx_3d).float()

        return evt.name, data
