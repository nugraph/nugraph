from abc import ABC
from typing import Any, Dict, List, Tuple

from ..io import File

class ProcessorBase(ABC):
    '''Base class for event processing'''

    def __init__(self, file: File):
        for group, keys in self.columns.items():
            file.add_group(group, keys)

    @property
    def columns(self) -> Dict[str, List[str]]:
        raise NotImplementedError

    def __call__(self, evt: Any) -> Tuple[str, Any]:
        raise NotImplementedError