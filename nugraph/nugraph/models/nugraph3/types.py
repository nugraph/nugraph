"""Convenient type aliases"""
from torch import Tensor as T
from torch_geometric.data import HeteroData, Batch
TD = dict[str, T] # tensor dictionary
TDD = dict[str, TD] # double nested tensor dictionary
EK = tuple[str, str, str] # edge key
ED = dict[EK, T] # edge dictionary
Data = HeteroData | Batch # graph data object

# data interface labels
N_IT = "particle-truth" # true instance node store
N_IP = "particle" # predicted instance node store
E_H_IT = ("hit", "cluster-truth", N_IT) # hit to true instance edges
E_H_IP = ("hit", "cluster", N_IP) # hit to predicted instance edges
