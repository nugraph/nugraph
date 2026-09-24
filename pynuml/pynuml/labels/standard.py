import pandas as pd
import particle


class StandardLabels:
    """Assign semantic and instance labels to simulated particles.

    The final semantic class, ``invisible``, is used internally for neutral
    particles that leave the simulated/active volume without a relevant
    interaction. The graph producer may later remap that class to ``-1`` if
    invisible activity is excluded from training.
    """

    TRANSPORT_PROCESSES = {
        "Transportation",
        "CoupledTransportation",
    }

    def __init__(
        self,
        gamma_threshold: float = 0.02,
        hadron_threshold: float = 0.2,
    ):
        self._labels = [
            "pion",
            "muon",
            "kaon",
            "hadron",
            "shower",
            "michel",
            "diffuse",
            "invisible",
        ]
        self._gamma_threshold = gamma_threshold
        self._hadron_threshold = hadron_threshold

    @property
    def labels(self):
        return self._labels

    def label(self, idx: int):
        """Return the class name associated with a semantic-label index."""
        if not 0 <= idx < len(self._labels):
            raise IndexError(
                f"index {idx} out of range for {len(self._labels)} labels."
            )
        return self._labels[idx]

    def index(self, name: str):
        """Return the semantic-label index associated with a class name."""
        if name not in self._labels:
            raise ValueError(f'"{name}" is not the name of a class.')
        return self._labels.index(name)

    @property
    def pion(self):
        return self.index("pion")

    @property
    def muon(self):
        return self.index("muon")

    @property
    def kaon(self):
        return self.index("kaon")

    @property
    def hadron(self):
        return self.index("hadron")

    @property
    def shower(self):
        return self.index("shower")

    @property
    def michel(self):
        return self.index("michel")

    @property
    def diffuse(self):
        return self.index("diffuse")

    @property
    def invisible(self):
        return self.index("invisible")

    @staticmethod
    def _normalise_process(value) -> str:
        """Convert a process value to a clean string for reliable matching."""
        if pd.isna(value):
            return ""
        return str(value).strip()

    def __call__(self, part: pd.DataFrame):
        """Apply the standard particle-labeling scheme.

        Classes are pion, muon, kaon, hadron, EM shower, Michel electron,
        diffuse activity, and invisible activity.
        """

        def walk(current, particles, depth, inherited_semantic, inherited_instance):
            del depth  # Kept in the signature for compatibility/debugging.

            def semantic_label(current, particles):
                semantic = -1
                child_semantic = None

                start_process = self._normalise_process(current.start_process)
                end_process = self._normalise_process(current.end_process)

                if current.parent_id == 0:
                    parent_type = 0
                else:
                    try:
                        parent_type = particles.at[current.parent_id, "type"]
                    except KeyError as exc:
                        raise RuntimeError(
                            f"parent particle {current.parent_id} was not found for "
                            f"particle {current.g4_id}."
                        ) from exc

                pdg_code = int(current.type)
                abs_pdg = abs(pdg_code)
                charge = particle.pdgid.charge(pdg_code)

                def pion_labeler(_current, _parent_type):
                    return self.pion, None

                def muon_labeler(_current, _parent_type):
                    return self.muon, None

                def kaon_labeler(_current, _parent_type):
                    return self.kaon, None

                def neutral_pions_kaons_labeler(_current, _parent_type):
                    return self.invisible, None

                def electron_positron_labeler(current, parent_type):

                    start_process = self._normalise_process(
                        current.start_process
                    )
                    end_process = self._normalise_process(
                        current.end_process
                    )
                
                    # Generator-level electron/positron.
                    if start_process.startswith("primary"):
                        return self.shower, self.shower
                
                    # Muon capture related electron/positron.
                    if (
                        abs(parent_type) == 13
                        and start_process in {
                            "muMinusCaptureAtRest",
                            "muPlusCaptureAtRest",
                        }
                    ):
                        return self.michel, self.michel
                
                    # Decay electron/positron.
                    if start_process == "Decay":
                        if abs(parent_type) == 13:
                            # True Michel electron/positron.
                            return self.michel, self.michel
                
                        # Electron/positron from another decay.
                        return self.shower, self.shower
                
                    # Photon conversion / Compton scattering.
                    if (
                        start_process in {"conv", "compt"}
                        or end_process in {"conv", "compt"}
                    ):
                        if current.momentum >= self._gamma_threshold:
                            return self.shower, self.shower
                
                        return self.diffuse, self.diffuse
                
                    # Ionization secondaries.
                    if start_process in {
                        "muIoni",
                        "hIoni",
                        "eIoni",
                    }:
                        if start_process == "muIoni":
                            return self.muon, None
                
                        if start_process == "hIoni":
                            if abs(parent_type) == 2212:
                                label = self.hadron
                
                                if current.momentum <= 0.0015:
                                    label = self.diffuse
                            else:
                                label = self.pion
                
                            return label, None
                
                        return self.diffuse, None
                
                    # Other secondary EM activity.
                    if (
                        start_process == "eBrem"
                        or end_process in {
                            "phot",
                            "photonNuclear",
                            "eIoni",
                        }
                    ):
                        return self.diffuse, None
                
                    if (
                        end_process in {
                            "StepLimiter",
                            "annihil",
                            "eBrem",
                            "FastScintillation",
                        }
                        or start_process in {
                            "hBertiniCaptureAtRest",
                            "muPairProd",
                            "phot",
                        }
                    ):
                        return self.diffuse, self.diffuse
                
                    raise RuntimeError(
                        "labelling failed for electron/positron: "
                        f"g4_id={current.g4_id}, "
                        f"parent_id={current.parent_id}, "
                        f"parent_pdg={parent_type}, "
                        f"pdg={current.type}, "
                        f"momentum={current.momentum}, "
                        f'start_process="{start_process}", '
                        f'end_process="{end_process}".'
                    )
                def gamma_labeler(_current, _parent_type):
                    if (
                        start_process in {"conv", "compt"}
                        or end_process in {"conv", "compt"}
                    ):
                        if current.momentum >= self._gamma_threshold:
                            return self.shower, self.shower
                        return self.diffuse, self.diffuse

                    if (
                        start_process == "eBrem"
                        or end_process in {"phot", "photonNuclear"}
                    ):
                        return self.diffuse, None

                    raise RuntimeError(
                        "labelling failed for photon with "
                        f'start process "{start_process}" and '
                        f'end process "{end_process}".'
                    )

                def unlabeled_particle():
                    raise RuntimeError(
                        "particle not recognised! "
                        f"PDG code {pdg_code}, parent PDG code {parent_type}, "
                        f'start process "{start_process}", '
                        f'end process "{end_process}".'
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
                    411: kaon_labeler,  # Existing grouping retained.
                    11: electron_positron_labeler,
                    22: gamma_labeler,
                }

                # Boundary-exit classification has the highest priority.
                # This handles, for example, a photon produced by pi+Inelastic
                # whose end process is Transportation. Because the photon left
                # the active/simulated volume, it is classified as invisible
                # instead of being sent to gamma_labeler and raising an error.
                neutral_transport = (
                    charge == 0 and end_process in self.TRANSPORT_PROCESSES
                )

                if neutral_transport:
                    semantic = self.invisible
                    child_semantic = None
                else:
                    processor = particle_processor.get(abs_pdg)
                    if processor is not None:
                        semantic, child_semantic = processor(current, parent_type)

                    # These rules are deliberately inside the `else` branch.
                    # Therefore, a neutral baryon/nucleus that leaves by a
                    # transport process remains invisible and is not later
                    # overwritten as diffuse.
                    if (
                        particle.pdgid.is_baryon(pdg_code) and charge == 0
                    ) or particle.pdgid.is_nucleus(pdg_code):
                        semantic = self.diffuse
                        child_semantic = None

                    elif particle.pdgid.is_baryon(pdg_code) and charge != 0:
                        if (
                            abs_pdg == 2212
                            and current.momentum >= self._hadron_threshold
                        ):
                            semantic = self.hadron
                        else:
                            semantic = self.diffuse
                        child_semantic = None

                    # Existing charged-tau convention retained.
                    if abs_pdg == 15:
                        semantic = self.hadron
                        child_semantic = None

                if semantic == -1:
                    unlabeled_particle()

                return semantic, child_semantic

            def instance_label(current, semantic):
                instance = -1
                child_instance = None
                start_process = self._normalise_process(current.start_process)

                if semantic == self.muon and start_process == "muIoni":
                    instance = current.parent_id
                elif (
                    semantic in {self.pion, self.hadron}
                    and start_process == "hIoni"
                ):
                    instance = current.parent_id
                elif semantic not in {self.diffuse, self.invisible}:
                    instance = current.g4_id
                    if semantic in {self.shower, self.michel}:
                        child_instance = instance

                return instance, child_instance

            if inherited_semantic is not None:
                semantic = inherited_semantic
                child_semantic = inherited_semantic
            else:
                semantic, child_semantic = semantic_label(current, particles)

            if inherited_instance is not None:
                instance = inherited_instance
                child_instance = inherited_instance
            else:
                instance, child_instance = instance_label(current, semantic)

            result = [
                {
                    "g4_id": current.g4_id,
                    "parent_id": current.parent_id,
                    "type": current.type,
                    "start_process": current.start_process,
                    "end_process": current.end_process,
                    "momentum": current.momentum,
                    "semantic_label": semantic,
                    "instance_label": instance,
                }
            ]

            children = particles[current.g4_id == particles.parent_id]
            for _, child in children.iterrows():
                result += walk(
                    child,
                    particles,
                    0,
                    child_semantic,
                    child_instance,
                )

            return result

        if part.empty:
            return None

        particles = part.set_index("g4_id", drop=False)
        primaries = particles[particles.parent_id == 0]

        records = []
        for _, primary in primaries.iterrows():
            records += walk(primary, particles, 0, None, None)

        if not records:
            return None

        labels = pd.DataFrame.from_dict(records)

        valid_instances = labels.loc[
            labels.instance_label >= 0, "instance_label"
        ].unique()
        instance_aliases = {
            value: index for index, value in enumerate(valid_instances)
        }

        labels["instance_label"] = labels["instance_label"].map(
            lambda value: -1 if value == -1 else instance_aliases[value]
        )

        return labels

    def validate(self, labels: pd.Series):
        """Validate labels produced by this class, including `invisible`."""
        mask = (labels < 0) | (labels >= len(self._labels))
        if mask.any():
            raise ValueError(
                f"{mask.sum()} semantic labels are out of range: "
                f"{labels[mask].tolist()}."
            )
