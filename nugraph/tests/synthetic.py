"""Synthetic NuGraph3 events for unit and regression tests

These mimic the graph layout written by pynuml's HitGraphProducer (gen 2:
merged "hit" store, delaunay-planar and nexus edges, an event node, and
particle-truth nodes), so tests run without any input file.
"""
import torch
from pynuml.data import NuGraphData

PLANES = ("u", "v", "y")
SEMANTIC_CLASSES = ("MIP", "HIP", "shower", "michel", "diffuse")
EVENT_CLASSES = ("numu", "nue", "nc")
IN_FEATURES = 8 # pos (2) + integral, rms, tpc, plane, proj, drift


def make_event(seed: int = 0, hits_per_plane: int = 30, n_particles: int = 4,
               n_spacepoints: int = 20, n_background: int = 4) -> NuGraphData:
    """
    Build one synthetic event in the gen-2 NuGraph format

    Each plane holds `hits_per_plane` hits. The last `n_background` hits in
    each plane are background (y_semantic == -1, no particle); the rest are
    split into `n_particles` contiguous blocks, one per true particle.

    Args:
        seed: Random seed, so the same seed always gives the same event
        hits_per_plane: Number of hits in each wire plane
        n_particles: Number of true particles
        n_spacepoints: Number of 3D spacepoints
        n_background: Number of background hits per plane
    """
    g = torch.Generator().manual_seed(seed)
    n_planes = len(PLANES)
    n_hit = n_planes * hits_per_plane

    data = NuGraphData()

    # hit nodes
    plane = torch.arange(n_planes).repeat_interleave(hits_per_plane)
    proj = torch.rand(n_hit, generator=g) * 100.
    drift = torch.rand(n_hit, generator=g) * 100.
    integral = torch.rand(n_hit, generator=g) * 50. + 10.
    rms = torch.rand(n_hit, generator=g) * 2. + 1.
    tpc = torch.zeros(n_hit)
    h = data["hit"]
    h.plane = plane
    h.pos = torch.stack((proj, drift), dim=1)
    h.x = torch.stack((integral, rms, tpc, plane.float(), proj, drift), dim=1)
    h.id = torch.arange(n_hit)
    h.y_position = torch.rand(n_hit, 3, generator=g) * 100.

    # truth: particle blocks within each plane, background at the end
    n_signal = hits_per_plane - n_background
    local = torch.arange(hits_per_plane).repeat(n_planes)
    signal = local < n_signal
    particle = torch.where(signal, local * n_particles // n_signal,
                           torch.full_like(local, -1))
    semantic = torch.where(signal, particle % (len(SEMANTIC_CLASSES) - 1),
                           torch.full_like(local, -1))
    h.y_semantic = semantic.long()

    pt = data["particle-truth"]
    pt.num_nodes = n_particles
    pt.momentum = torch.rand(n_particles, generator=g)
    pt.pdg_code = torch.full((n_particles,), 13, dtype=torch.long)
    pt.g4_id = torch.arange(1, n_particles + 1)
    hit_idx = torch.nonzero(signal).squeeze(1)
    data["hit", "cluster-truth", "particle-truth"].edge_index = torch.stack(
        (hit_idx, particle[signal]), dim=0)

    # planar edges: a chain plus skip connections within each plane
    edges = []
    for p in range(n_planes):
        idx = torch.arange(p * hits_per_plane, (p + 1) * hits_per_plane)
        for step in (1, 2):
            e = torch.stack((idx[:-step], idx[step:]), dim=0)
            edges += [e, e.flip(0)]
    data["hit", "delaunay-planar", "hit"].edge_index = torch.cat(edges, dim=1)

    # spacepoints: each joins one random hit from every plane
    data["sp"].pos = torch.rand(n_spacepoints, 3, generator=g) * 100.
    sp_hits = torch.stack(
        [torch.randint(p * hits_per_plane, (p + 1) * hits_per_plane,
                       (n_spacepoints,), generator=g) for p in range(n_planes)],
        dim=1)
    sp_idx = torch.arange(n_spacepoints).repeat_interleave(n_planes)
    data["hit", "nexus", "sp"].edge_index = torch.stack(
        (sp_hits.flatten(), sp_idx), dim=0)

    # event node
    data["evt"].num_nodes = 1
    data["evt"].y = torch.tensor(seed % len(EVENT_CLASSES))
    data["evt"].y_vtx = torch.rand(1, 3, generator=g) * 100.
    data["hit", "in", "evt"].edge_index = torch.stack(
        (torch.arange(n_hit), torch.zeros(n_hit, dtype=torch.long)), dim=0)
    data["sp", "in", "evt"].edge_index = torch.stack(
        (torch.arange(n_spacepoints),
         torch.zeros(n_spacepoints, dtype=torch.long)), dim=0)

    return data


def make_batch(*seeds: int, **kwargs) -> "torch_geometric.data.Batch":
    """Build a transformed batch of synthetic events, one per seed"""
    # pylint: disable=import-outside-toplevel
    from torch_geometric.data import Batch
    from nugraph.models.nugraph3.transform import Transform
    transform = Transform(PLANES)
    return Batch.from_data_list(
        [transform(make_event(s, **kwargs)) for s in seeds])
