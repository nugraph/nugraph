"""Convenient type aliases"""
from torch import Tensor as T
TD = dict[str, T] # tensor dictionary
EK = tuple[str, str, str] # edge key
ED = dict[EK, T] # edge dictionary
