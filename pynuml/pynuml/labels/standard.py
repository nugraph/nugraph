import pandas as pd
import particle

class StandardLabels:

    def __init__(self,
                 gamma_threshold: float = 0.02,
                 hadron_threshold: float = 0.2):
        self._labels = [
            'pion',
            'muon',
            'kaon',
            'hadron',
            'shower',
            'michel',
            'diffuse',
            'invisible'
        ]
        self._gamma_threshold = gamma_threshold
        self._hadron_threshold = hadron_threshold

    @property
    def labels(self):
        return self._labels

    def label(self, idx: int):
        if not 0 <= label < len(self._labels):
            raise Exception(f'index {idx} out of range for {len(self._labels)} labels.')
        return self._labels[idx]

    def index(self, name: str):
        if name not in self._labels:
            raise Exception(f'"{name}" is not the name of a class.')
        return self._labels.index(name)
    
    @property
    def pion(self):
        return self.index('pion')

    @property
    def muon(self):
        return self.index('muon')

    @property
    def kaon(self):
        return self.index('kaon')

    @property
    def hadron(self):
        return self.index('hadron')

    @property
    def shower(self):
        return self.index('shower')

    @property
    def michel(self):
        return self.index('michel')

    @property
    def diffuse(self):
        return self.index('diffuse')
    
    @property
    def invisible(self):
        return self.index('invisible')

    def __call__(self,
                 part: pd.DataFrame):
        '''Standard labelling function.

        Pion, Muon, Kaon, Hadron, EM shower, Michel electron,
        diffuse activity.
        '''

        def walk(part, particles, depth, sl, il):
            def s(part, particles):
                sl, slc = -1, None
                parent_type = 0 if part.parent_id == 0 else particles.type[part.parent_id]

                def pion_labeler(part, parent_type):
                    sl = self.pion
                    slc = None
                    return sl, slc

                def muon_labeler(part, parent_type):
                    sl = self.muon
                    slc = None
                    return sl, slc

                def kaon_labeler(part, parent_type):
                    sl = self.kaon
                    slc = None
                    return sl, slc

                def neutral_pions_kaons_labeler(part, parent_type):
                    sl = self.invisible
                    slc = None
                    return sl, slc

                def electron_positron_labeler(part, parent_type):
                    if part.start_process == 'primary':
                        sl = self.shower
                        slc = self.shower
                    elif abs(parent_type) == 13 and (part.start_process == 'muMinusCaptureAtRest' \
                        or part.start_process == 'muPlusCaptureAtRest' or part.start_process == 'Decay'):
                        sl = self.michel
                        slc = self.michel
                    elif part.start_process == 'conv' or part.end_process == 'conv' \
                        or part.start_process == 'compt' or part.end_process == 'compt':
                        if part.momentum >= self._gamma_threshold:
                            sl = self.shower
                            slc = self.shower
                        else:
                            sl = self.diffuse
                            slc = self.diffuse
                    elif part.start_process == 'muIoni' or part.start_process == 'hIoni' \
                        or part.start_process == 'eIoni':
                        if part.start_process == 'muIoni':
                            sl = self.muon
                            slc = None
                        elif part.start_process == 'hIoni':
                            if abs(parent_type) == 2212:
                                sl = self.hadron
                                if part.momentum <= 0.0015: sl = self.diffuse
                            else:
                                sl = self.pion
                            slc = None
                        else:
                            sl = self.diffuse
                            slc = None
                    elif part.start_process == 'eBrem' or part.end_process == 'phot' \
                        or part.end_process == 'photonNuclear' or part.end_process == 'eIoni':
                        sl = self.diffuse
                        slc = None
                    elif part.end_process == 'StepLimiter' or part.end_process == 'annihil' \
                        or part.end_process == 'eBrem' or part.start_process == 'hBertiniCaptureAtRest' \
                        or part.end_process == 'FastScintillation' or part.start_process == 'muPairProd' \
                        or part.start_process == 'phot':
                        sl = self.diffuse
                        slc = self.diffuse
                    else:
                        raise Exception(f'labelling failed for electron with start process "{part.start_process}" and end process "{part.end_process}')

                    return sl, slc

                def gamma_labeler(part, parent_type):
                    if part.start_process == 'conv' or part.end_process == 'conv' \
                        or part.start_process == 'compt' or part.end_process == 'compt':
                        if part.momentum >= self._gamma_threshold:
                            sl = self.shower
                            slc = self.shower
                        else:
                            sl = self.diffuse
                            slc = self.diffuse
                    elif part.start_process == 'eBrem' or part.end_process == 'phot' \
                        or part.end_process == 'photonNuclear':
                        sl = self.diffuse
                        slc = None
                    else:
                        raise Exception(f'labelling failed for photon with start process "{part.start_process}" and end process "{part.end_process}')
                    return sl, slc

                def unlabeled_particle(part, parent_type):
                    raise Exception(f"particle not recognised! PDG code {part.type}, parent PDG code {parent_type}, start process {part.start_process}, end process {part.end_process}")

                particle_processor = {
                    211: pion_labeler,
                    221: pion_labeler,
                    331: pion_labeler,
                    223: pion_labeler,
                    13: muon_labeler,
                    321: kaon_labeler,
                    111: neutral_pions_kaons_labeler,
                    311: neutral_pions_kaons_labeler,
                    310: neutral_pions_kaons_labeler,
                    130: neutral_pions_kaons_labeler,
                    113: neutral_pions_kaons_labeler,
                    411: kaon_labeler, # D meson
                    11: electron_positron_labeler,
                    22: gamma_labeler
                }

                if particle.pdgid.charge(part.type) == 0 and part.end_process == 'CoupledTransportation':
                    # neutral particle left the volume boundary
                    sl = self.invisible
                else:
                    func = particle_processor.get(abs(part.type), lambda x ,y: (-1, None))
                    sl, slc = func(part, parent_type)

                # baryon interactions - hadron or diffuse
                if (particle.pdgid.is_baryon(part.type) and particle.pdgid.charge(part.type) == 0) \
                    or particle.pdgid.is_nucleus(part.type):
                    sl = self.diffuse
                if particle.pdgid.is_baryon(part.type) and particle.pdgid.charge(part.type) != 0:
                    if abs(part.type) == 2212 and part.momentum >= self._hadron_threshold:
                        sl = self.hadron
                    else:
                        sl = self.diffuse

                # call a charged tau highly ionising - should revisit this
                if abs(part.type) == 15:
                    sl = self.hadron

                # check to make sure particle was assigned
                if sl == -1:
                    unlabeled_particle(part, parent_type)

                return sl, slc

            def i(part, particles, sl):
                il, ilc = -1, None
                if sl == self.muon and part.start_process == 'muIoni':
                    il = part.parent_id
                elif (sl == self.pion or sl == self.hadron) and part.start_process == 'hIoni':
                    il = part.parent_id
                elif sl != self.diffuse and sl != self.invisible:
                    il = part.g4_id
                    if sl == self.shower: ilc = il
                    if sl == self.michel: ilc = il
                return il, ilc

            if sl is not None: slc = sl
            else: sl, slc = s(part, particles)

            if il is not None: ilc = il
            else: il, ilc = i(part, particles, sl)

            ret = [ {
                "g4_id": part.g4_id,
                "parent_id": part.parent_id,
                "type": part.type,
                "start_process": part.start_process,
                "end_process": part.end_process,
                "momentum": part.momentum,
                "semantic_label": sl,
                "instance_label": il } ]
            for _, row in particles[(part.g4_id==particles.parent_id)].iterrows():
                ret += walk(row, particles, depth+1, slc, ilc)
            return ret

        ret = []
        part = part.set_index("g4_id", drop=False)
        primaries = part[(part.parent_id==0)]
        for _, primary in primaries.iterrows():
            ret += walk(primary, part, 0, None, None)
        if len(ret)==0: return
        labels = pd.DataFrame.from_dict(ret)
        instances = { val: i for i, val in enumerate(labels[(labels.instance_label>=0)].instance_label.unique()) }

        def alias_instance(row, instances):
            if row.instance_label == -1: return -1
            return instances[row.instance_label]

        labels["instance_label"] = labels.apply(alias_instance, args=[instances], axis="columns")
        return labels

    def validate(self, labels: pd.Series):
        mask = (labels < 0) | (labels >= len(self._labels) - 1)
        if mask.any():
            raise Exception(f'{mask.sum()} semantic labels are out of range: {labels[mask]}.')
