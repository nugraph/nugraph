"""Test pynuml semantic labelling"""
import pandas as pd
import pynuml

def particles(rows):
    """Build a particle table from rows of
    (g4_id, parent_id, type, start_process, end_process, momentum)"""
    return pd.DataFrame(rows, columns=["g4_id", "parent_id", "type", "start_process",
                                       "end_process", "momentum"])

def labels(labeller, rows):
    """Semantic label of each particle, by g4_id"""
    ret = labeller(particles(rows))
    return dict(zip(ret.g4_id, ret.semantic_label))

def test_decay_electrons():
    """Electrons from muon decays are Michel electrons, from neutral pion Dalitz decays showers"""
    l = pynuml.labels.StandardLabels()
    sl = labels(l, [
        (1, 0, 13, "primary", "Decay", 0.3),
        (2, 1, 11, "Decay", "eIoni", 0.04),
        (3, 0, 111, "primary", "Decay", 0.3),
        (4, 3, 11, "Decay", "eIoni", 0.05),
        (5, 3, -11, "Decay", "annihil", 0.05),
        (6, 4, 22, "eBrem", "phot", 0.001)])
    assert sl[2] == l.michel
    assert sl[4] == l.shower
    assert sl[5] == l.shower
    assert sl[6] == l.shower

def test_photon_leaving_world():
    """Photons leaving the world through Transportation are diffuse"""
    l = pynuml.labels.StandardLabels()
    sl = labels(l, [
        (1, 0, 2112, "primary", "nCapture", 0.1),
        (2, 1, 22, "nCapture", "CoupledTransportation", 0.005),
        (3, 1, 22, "nCapture", "Transportation", 0.005)])
    assert sl[2] == l.invisible
    assert sl[3] == l.diffuse

def test_beam_background():
    """Beam-line background electrons are showers only for test beam labelling"""
    rows = [(1, 0, -11, "primaryBackground", "annihil", 0.5)]
    l = pynuml.labels.StandardLabels()
    assert labels(l, rows)[1] == l.diffuse
    l = pynuml.labels.StandardLabels(beam_background=True)
    assert labels(l, rows)[1] == l.shower
