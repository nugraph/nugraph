def ccqe(part):
    # get primary for each particle
    part = part.set_index("g4_id", drop=False)

    # convert from PDG code to label
    def label(pdg):
        if abs(pdg) == 11: return 0 # electron
        if abs(pdg) == 13: return 1 # muon
        return 2 # hadronic

    # trace lineage back from particle to primary and get label
    def func(row):
        gid = row.g4_id
        pid = row.parent_id
        while True:
            if pid == 0: return label(part.type[gid])
            # if not pid in part.g4_id: return label(part.type[gid])
            # gid = part.g4_id[pid]
            try:
                gid = part.g4_id[pid]
            except KeyError:
                return 2
            pid = part.parent_id[pid]

    # apply backtrace function to get labels
    part["semantic_label"] = part.apply(func, axis=1)
    return part.reset_index(drop=True)[["g4_id", "semantic_label"]]

def panoptic_label(part):
    part = semantic_label(part)
    part["instance_label"] = -1
    return part

def edge_label(edge):

    # False
    edge["label"] = 0

    # EM shower
    mask_e = (edge.label_1 == 0) & (edge.label_2 == 0)
    edge.loc[mask_e, "label"] = 1

    # Muon
    mask_part = (edge.g4_id_1 == edge.g4_id_2)
    mask_mu = (edge.label_1 == 1) & (edge.label_2 == 1)
    edge.loc[mask_part & mask_mu, "label"] = 2

    # Hadronic
    mask_had = (edge.label_1 == 2) & (edge.label_2 == 2)
    edge.loc[mask_part & mask_had, "label"] = 3

    return edge
