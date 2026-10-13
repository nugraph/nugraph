"""Tests for the wire-geometry encoding of hit planes (--plane-geometry)"""
import pytest
import torch

from nugraph.models.nugraph3.nugraph3 import plane_geometry

from synthetic import PLANES
from test_nexus_decoder import nexus_batch, nexus_model

# synthetic hits are (integral, rms, tpc, plane, proj, drift): wire coordinate in column 4, plane in 3
GEOMETRY = [[-0.866, 0.5, 101.4], [0.866, 0.5, 99.8], [0., 1., -0.3]]
COLUMNS = (4, 3)


def test_plane_index_becomes_wire_geometry():
    """The plane index is replaced by the plane's wire direction, and the wire
    coordinate is measured from the origin; the other features pass unchanged"""
    model = nexus_model(plane_geometry=GEOMETRY, geometry_columns=COLUMNS)
    batch = nexus_batch(0)
    x, plane = batch["hit"].x.clone(), batch["hit"].plane
    seen = {}
    norm = model.tpc_encoder.input_norm.forward
    model.tpc_encoder.input_norm.forward = lambda t: (seen.setdefault("x", t), norm(t))[1]
    model.eval()
    with torch.no_grad():
        model(batch)
    g = torch.tensor(GEOMETRY)[plane]
    expected = torch.cat((x[:, :3], (x[:, 4] - g[:, 2])[:, None], x[:, 5:], g[:, :2]), dim=1)
    assert torch.allclose(seen["x"], expected)
    assert model.tpc_encoder.input_norm.norm["mean"].numel() == x.size(1) + 1


def test_plane_geometry_training_step():
    """A training step with the geometry encoding gives a finite loss"""
    model = nexus_model(plane_geometry=GEOMETRY, geometry_columns=COLUMNS)
    model.train()
    loss, _ = model(nexus_batch(0, 1), stage="train")
    loss.backward()
    assert torch.isfinite(loss)


def test_plane_geometry_argument():
    """--plane-geometry takes three numbers per plane"""
    assert plane_geometry(None, PLANES) is None
    assert plane_geometry(sum(GEOMETRY, []), PLANES) == GEOMETRY
    with pytest.raises(RuntimeError, match="three numbers"):
        plane_geometry([0., 1., 0.], PLANES)
