import pandas as pd
import particle


class StandardLabels:

    def __init__(self,
                 gamma_threshold: float = 0.02,
                 hadron_threshold: float = 0.2,
                 strict: bool = False):
        """
        strict=False: unknown cases fall back to diffuse/shower instead of raising.
        strict=True: preserves old behavior by raising on unknown processes.
        """
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
        self._strict = strict

    @property
    def labels(self):
        return self._labels

    def label(self, idx: int):
        if not 0 <= idx < len(self._labels):
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

    def __call__(self, part: pd.DataFrame):
        """
        Input: particle_table dataframe with columns including:
          g4_id, parent_id, type, start_process, end_process, momentum
        Output: dataframe with semantic_label and instance_label for each particle in the truth tree
        """

        if part is None or len(part) == 0:
            return

        # ensure g4_id index
        part = part.set_index("g4_id", drop=False)

        # If g4_id duplicates exist, keep first to guarantee scalar lookups.
        if not part.index.is_unique:
            part = part[~part.index.duplicated(keep="first")]

        def _proc_str(x) -> str:
            # Normalize process strings: handle None, strip whitespace/newlines
            if x is None:
                return ""
            try:
                return str(x).strip()
            except Exception:
                return ""

        def _scalar_parent_type(particles: pd.DataFrame, parent_id: int) -> int:
            """
            Robustly return scalar PDG code for parent particle.
            Avoids pandas Series being returned (which breaks boolean logic).
            """
            if parent_id == 0:
                return 0

            # best case: index is g4_id
            if parent_id in particles.index:
                v = particles.at[parent_id, "type"]
                if isinstance(v, pd.Series):
                    v = v.iloc[0]
                try:
                    return int(v)
                except Exception:
                    return 0

            # fallback: search by column (in case index differs)
            try:
                sel = particles.loc[particles["g4_id"] == parent_id, "type"]
                if len(sel) == 0:
                    return 0
                return int(sel.iloc[0])
            except Exception:
                return 0

        def _raise_or_fallback(msg: str, fallback_sl: int, fallback_slc):
            if self._strict:
                raise Exception(msg)
            return fallback_sl, fallback_slc

        def walk(p, particles: pd.DataFrame, depth: int, sl, il):
            def s(particle_row, particles_df: pd.DataFrame):
                sl_local, slc_local = -1, None

                parent_id = int(particle_row.parent_id) if pd.notna(particle_row.parent_id) else 0
                parent_type = _scalar_parent_type(particles_df, parent_id)

                sp = _proc_str(particle_row.start_process)
                ep = _proc_str(particle_row.end_process)

                def pion_labeler(_row, _parent_type):
                    return self.pion, None

                def muon_labeler(_row, _parent_type):
                    return self.muon, None

                def kaon_labeler(_row, _parent_type):
                    return self.kaon, None

                def neutral_pions_kaons_labeler(_row, _parent_type):
                    return self.invisible, None

                def electron_positron_labeler(_row, parent_type_):
                    sp2 = _proc_str(_row.start_process)
                    ep2 = _proc_str(_row.end_process)

                    if sp2 == 'primary':
                        return self.shower, self.shower

                    # Michel electrons from muon capture/decay
                    if abs(parent_type_) == 13 and (
                        sp2 in ('muMinusCaptureAtRest', 'muPlusCaptureAtRest', 'Decay')
                    ):
                        return self.michel, self.michel

                    # conversion / compton
                    if sp2 in ('conv', 'compt') or ep2 in ('conv', 'compt'):
                        if float(_row.momentum) >= self._gamma_threshold:
                            return self.shower, self.shower
                        return self.diffuse, self.diffuse

                    # ionization products
                    if sp2 in ('muIoni', 'hIoni', 'eIoni'):
                        if sp2 == 'muIoni':
                            return self.muon, None
                        if sp2 == 'hIoni':
                            if abs(parent_type_) == 2212:
                                slx = self.hadron
                                try:
                                    if float(_row.momentum) <= 0.0015:
                                        slx = self.diffuse
                                except Exception:
                                    pass
                                return slx, None
                            return self.pion, None
                        return self.diffuse, None

                    # various low-energy / diffuse electron cases
                    if sp2 == 'eBrem' or ep2 in ('phot', 'photonNuclear', 'eIoni'):
                        return self.diffuse, None

                    if ep2 in ('StepLimiter', 'annihil', 'eBrem', 'FastScintillation') \
                       or sp2 in ('hBertiniCaptureAtRest', 'muPairProd', 'phot'):
                        return self.diffuse, self.diffuse

                    # fallback (instead of hard crash)
                    return _raise_or_fallback(
                        f'labelling failed for electron with start process "{sp2}" and end process "{ep2}"',
                        self.diffuse, self.diffuse
                    )

                def gamma_labeler(_row, parent_type_):
                    sp2 = _proc_str(_row.start_process)
                    ep2 = _proc_str(_row.end_process)

                    # EM interactions
                    if sp2 in ('conv', 'compt') or ep2 in ('conv', 'compt'):
                        if float(_row.momentum) >= self._gamma_threshold:
                            return self.shower, self.shower
                        return self.diffuse, self.diffuse

                    # brem / photon-nuclear / generic photon production
                    if sp2 == 'eBrem' or ep2 in ('phot', 'photonNuclear'):
                        return self.diffuse, None

                    # photons from hadronic/transport processes (your failing case)
                    if sp2 in ('neutronInelastic', 'hadElastic', 'hadInelastic', 'nCapture', 'pi0Decay', 'Decay') \
                       or ep2 == 'Transportation':
                        if float(_row.momentum) >= self._gamma_threshold:
                            return self.shower, self.shower
                        return self.diffuse, self.diffuse

                    # fallback (instead of hard crash)
                    if float(_row.momentum) >= self._gamma_threshold:
                        return self.shower, self.shower
                    return self.diffuse, self.diffuse

                def unlabeled_particle(_row, _parent_type):
                    raise Exception(
                        f"particle not recognised! PDG code {_row.type}, parent PDG code {_parent_type}, "
                        f"start process {sp}, end process {ep}"
                    )

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
                    411: kaon_labeler,  # D meson
                    11: electron_positron_labeler,
                    22: gamma_labeler,
                }

                # neutral particle left boundary
                if particle.pdgid.charge(particle_row.type) == 0 and ep == 'CoupledTransportation':
                    sl_local = self.invisible
                else:
                    func = particle_processor.get(abs(particle_row.type), None)
                    if func is None:
                        sl_local, slc_local = -1, None
                    else:
                        sl_local, slc_local = func(particle_row, parent_type)

                # baryon interactions - hadron or diffuse
                try:
                    is_baryon = particle.pdgid.is_baryon(particle_row.type)
                    is_nucleus = particle.pdgid.is_nucleus(particle_row.type)
                    charge = particle.pdgid.charge(particle_row.type)
                except Exception:
                    is_baryon, is_nucleus, charge = False, False, 0

                if (is_baryon and charge == 0) or is_nucleus:
                    sl_local = self.diffuse

                if is_baryon and charge != 0:
                    if abs(particle_row.type) == 2212 and float(particle_row.momentum) >= self._hadron_threshold:
                        sl_local = self.hadron
                    else:
                        sl_local = self.diffuse

                # charged tau highly ionising - should revisit this
                if abs(particle_row.type) == 15:
                    sl_local = self.hadron

                if sl_local == -1:
                    if self._strict:
                        unlabeled_particle(particle_row, parent_type)
                    else:
                        sl_local = self.diffuse
                        slc_local = self.diffuse

                return sl_local, slc_local

            def i(particle_row, _particles, semantic_label):
                il_local, ilc_local = -1, None
                sp2 = _proc_str(particle_row.start_process)

                if semantic_label == self.muon and sp2 == 'muIoni':
                    il_local = particle_row.parent_id
                elif (semantic_label == self.pion or semantic_label == self.hadron) and sp2 == 'hIoni':
                    il_local = particle_row.parent_id
                elif semantic_label != self.diffuse and semantic_label != self.invisible:
                    il_local = particle_row.g4_id
                    if semantic_label == self.shower:
                        ilc_local = il_local
                    if semantic_label == self.michel:
                        ilc_local = il_local
                return il_local, ilc_local

            if sl is not None:
                slc = sl
            else:
                sl, slc = s(p, particles)

            if il is not None:
                ilc = il
            else:
                il, ilc = i(p, particles, sl)

            ret_local = [{
                "g4_id": p.g4_id,
                "parent_id": p.parent_id,
                "type": p.type,
                "start_process": _proc_str(p.start_process),
                "end_process": _proc_str(p.end_process),
                "momentum": p.momentum,
                "semantic_label": sl,
                "instance_label": il
            }]

            # recurse to children
            try:
                children = particles[(p.g4_id == particles.parent_id)]
                for _, row in children.iterrows():
                    ret_local += walk(row, particles, depth + 1, slc, ilc)
            except Exception:
                pass

            return ret_local

        # start with primaries
        ret = []
        primaries = part[(part.parent_id == 0)]
        for _, primary in primaries.iterrows():
            ret += walk(primary, part, 0, None, None)

        if len(ret) == 0:
            return

        labels = pd.DataFrame.from_dict(ret)

        # alias instance labels to compact range
        instances = {
            val: i for i, val in enumerate(labels[(labels.instance_label >= 0)].instance_label.unique())
        }

        def alias_instance(row):
            if row.instance_label == -1:
                return -1
            return instances.get(row.instance_label, -1)

        labels["instance_label"] = labels.apply(alias_instance, axis="columns")
        return labels

    def validate(self, labels: pd.Series):
        mask = (labels < 0) | (labels >= len(self._labels) - 1)
        if mask.any():
            raise Exception(f'{mask.sum()} semantic labels are out of range: {labels[mask]}.')
