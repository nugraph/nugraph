"""Test pynuml graph processing and plotting"""
import pynuml

def test_process_uboone():
    """Test graph processing with MicroBooNE open data release"""
    f = pynuml.io.File("/raid/nugraph/uboone-opendata/uboone-opendata.evt.h5")
    processor = pynuml.process.HitGraphProducer(
        file=f,
        semantic_labeller=pynuml.labels.StandardLabels(),
        event_labeller=pynuml.labels.FlavorLabels(),
        label_vertex=True)
    plot = pynuml.plot.GraphPlot(
        planes=["u", "v", "y"],
        classes=pynuml.labels.StandardLabels().labels[:-1])
    f.read_data(0, 100)
    evts = f.build_evt()
    for evt in evts:
        _, data = processor(evt)
        if not data:
            continue
        plot.plot(data, target='semantic', how='true', filter='show')
        plot.plot(data, target='instance', how='true', filter='true')

def test_process_dune_nutau():
    """Test graph processing with DUNE beam nutau dataset"""
    f = pynuml.io.File("/raid/nugraph/dune-nutau/test.evt.h5")
    processor = pynuml.process.HitGraphProducer(
        file=f,
        semantic_labeller=pynuml.labels.StandardLabels(),
        event_labeller=pynuml.labels.FlavorLabels(),
        label_position=True)
    plot = pynuml.plot.GraphPlot(
        planes=["u", "v", "y"],
        classes=pynuml.labels.StandardLabels().labels[:-1])
    f.read_data(0, 100)
    evts = f.build_evt()
    for evt in evts:
        _, data = processor(evt)
        if not data:
            continue
        plot.plot(data, target="filter", how="true", filter="show")
        plot.plot(data, target='semantic', how='true', filter='show')
        plot.plot(data, target='instance', how='true', filter='true')
        plot.plot(data, target="semantic", how="true", filter="show", xyz=True)
