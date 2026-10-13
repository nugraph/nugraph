"""Tests for the measurement-axis encoding of hit planes (--plane-geometry)"""
import json

import pytest
import torch

from nugraph.models.nugraph3.nugraph3 import plane_geometry

from synthetic import PLANES
from test_nexus_decoder import nexus_batch, nexus_model

# per plane, two axes of (direction, origin, resolution): wire coordinate, then drift
GEOMETRY = [[0., -0.8660254, 0.5, 101.4, 0.3, 1., 0., 0., 44., 0.055],
            [0., 0.8660254, 0.5, 99.8, 0.3, 1., 0., 0., 44., 0.055],
            [0., 0., 1., -0.3, 0.3, 1., 0., 0., 44., 0.055]]
# synthetic hits are (proj, drift, integral, rms, tpc, plane, proj, drift)
COLUMNS = (0, 1, 5)


def test_plane_index_becomes_measurement_axes():
    """The plane index is replaced by the plane's two axis directions and
    resolutions, and both coordinates are measured from the origin; the other
    features pass unchanged"""
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
    expected = torch.cat(((x[:, 0] - g[:, 3])[:, None], (x[:, 1] - g[:, 8])[:, None], x[:, 2:5], x[:, 6:],
                          g[:, [0, 1, 2, 4, 5, 6, 7, 9]]), dim=1)
    assert torch.allclose(seen["x"], expected)
    assert model.tpc_encoder.input_norm.norm["mean"].numel() == x.size(1) + 7


def test_plane_geometry_training_step():
    """A training step with the geometry encoding gives a finite loss"""
    model = nexus_model(plane_geometry=GEOMETRY, geometry_columns=COLUMNS)
    model.train()
    loss, _ = model(nexus_batch(0, 1), stage="train")
    loss.backward()
    assert torch.isfinite(loss)


def write(tmp_path, planes):
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps({"planes": planes}))
    return str(path)


def axes(row):
    return [{"direction": row[0:3], "origin": row[3], "resolution": row[4]},
            {"direction": row[5:8], "origin": row[8], "resolution": row[9]}]


def test_plane_geometry_file(tmp_path):
    """A geometry file gives ten numbers per plane, with unit directions; it
    must name the data's planes in order and give each two axes"""
    assert plane_geometry(None, PLANES) is None
    views = [{"name": p, "axes": axes(r)} for p, r in zip(PLANES, GEOMETRY)]
    assert torch.allclose(torch.tensor(plane_geometry(write(tmp_path, views), PLANES)),
                          torch.tensor(GEOMETRY))
    views[2]["axes"][0]["direction"] = [0., 0., 2.]
    assert plane_geometry(write(tmp_path, views), PLANES)[2][:3] == [0., 0., 1.]
    with pytest.raises(RuntimeError, match="describes planes"):
        plane_geometry(write(tmp_path, views[::-1]), PLANES)
    views[0]["axes"] = views[0]["axes"][:1]
    with pytest.raises(RuntimeError, match="two axes"):
        plane_geometry(write(tmp_path, views), PLANES)
