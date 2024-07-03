import pandas as pd

class PDKLabels:
    def __init__(self):
        self._labels = ('nu', 'pdk')

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
    def nu(self):
        return self.index('nu')

    @property
    def pdk(self):
        return self.index('pdk')

    def __call__(self, event: pd.Series):
        if 12 <= abs(event.nu_pdg) <= 16:
            return self.nu
        else:
            return self.pdk