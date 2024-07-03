"""Convenient type aliases"""
from torch import Tensor as T
TD = dict[str, T] # tensor dictionary
TDD = dict[str, TD] # double nested tensor dictionary
EK = tuple[str, str, str] # edge key
ED = dict[EK, T] # edge dictionary
