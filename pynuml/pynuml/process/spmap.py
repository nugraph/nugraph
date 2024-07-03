import pynuml

def process_event(key, out, sp, hit, part, edep, l=standard, voxelsize=1):
    """Process an event into a 3D pixel map"""
    import numpy as np, torch, MinkowskiEngine as ME

    # skip any events with no simulated hits
    if (hit.index==key).sum() == 0: return
    if (edep.index==key).sum() == 0: return

    # label true particles
    evt_part = part.loc[key].reset_index(drop=True)
    evt_part = l.panoptic_label(evt_part)

    # get energy depositions and ground truth
    evt_edep = edep.loc[key].reset_index(drop=True)
    evt_edep = evt_edep.merge(evt_part[["g4_id", "semantic_label"]], on="g4_id", how="left").drop("g4_id", axis="columns")
    scores = evt_edep.groupby(["hit_id", "semantic_label"]).agg({"energy": "sum"}).reset_index()

    # class number and names
    n = len(l.label) - 1
    lnames = [ it.name for it in l.label ][:-1]
    noise = np.zeros(n)
    noise[l.label.diffuse.value] = 1

    def fractional_truth(row, n):
        label = np.zeros(n)
        label[int(row.semantic_label)] = row.energy
        return label
    scores["slabel"] = scores.apply(fractional_truth, args=[n], axis="columns")
    scores = scores.groupby("hit_id").agg({"slabel": "sum"})

    # Propagate labels to hits
    evt_hit = hit.loc[key].reset_index(drop=True).merge(scores, on="hit_id", how="inner")
    evt_sp = sp.loc[key].reset_index(drop=True)

    # skip events with fewer than 50 simulated hits in any plane, or fewer than 50 spacepoints
    for i in range(3):
        if (evt_hit.global_plane==i).sum() < 50: return
    if evt_sp.shape[0] < 50: return

    # merge hits into spacepoints
    for plane in ["u","v","y"]:
        evt_sp = evt_sp.merge(evt_hit[["hit_id","integral","slabel"]].add_suffix(f"_{plane}"), on=f"hit_id_{plane}", how="left")
        evt_sp[f"integral_{plane}"] = evt_sp[f"integral_{plane}"].fillna(0)

    def merge_truth(row, n):
        labels = np.zeros(n)
        for plane in ["u","v","y"]:
            vals = row[f"slabel_{plane}"]
            if type(vals) != float: labels += vals
        return labels

    evt_sp["slabel"] = evt_sp.apply(merge_truth, args=[len(l.label)-1], axis="columns")
    evt_sp = evt_sp[["slabel", "position_x", "position_y", "position_z", "integral_u", "integral_v", "integral_y"]]

    # voxelise spacepoints and aggregate labels
    def voxelise(row):
        return np.floor(row.position_x/voxelsize), np.floor(row.position_y/voxelsize), np.floor(row.position_z/voxelsize)
    evt_sp["c"] = evt_sp.apply(voxelise, axis="columns")
    evt_sp = evt_sp.drop(["position_x", "position_y", "position_z"], axis="columns")
    evt_sp = evt_sp.groupby("c").agg({"integral_u": "sum", "integral_v": "sum", "integral_y": "sum", "slabel": "sum"}).reset_index()
    def norm_truth(row, noise):
        lsum = row.slabel.sum()
        return noise if lsum == 0 else row.slabel / lsum
    evt_sp["slabel"] = evt_sp.apply(norm_truth, args=[noise], axis="columns")

    spm = {
        "f": torch.tensor(evt_sp[["integral_u", "integral_v", "integral_y"]].to_numpy()).float(),
        "c": torch.tensor(evt_sp["c"]).int(),
        "ys": torch.tensor(evt_sp["slabel"]).float()
    }
    out.save(spm, f"r{key[0]}_sr{key[1]}_evt{key[2]}")

def process_file(out, fname, p=process_event, l=standard, voxelsize=1):
    """Process all events in a file into graphs"""
    f = NuMLFile(fname)

    evt = f.get_dataframe("event_table", ["event_id"])
    sp = f.get_dataframe("spacepoint_table")
    hit = f.get_dataframe("hit_table")
    part = f.get_dataframe("particle_table", ["event_id", "g4_id", "parent_id", "type", "momentum", "start_process", "end_process"])
    edep = f.get_dataframe("edep_table")

    # loop over events in file
    for key in evt.index: p(key, out, sp, hit, part, edep, l, voxelsize)

