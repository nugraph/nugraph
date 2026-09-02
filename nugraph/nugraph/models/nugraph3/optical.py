"""NuGraph optical convolution module"""
import torch
from pynuml.data import NuGraphData
from .core import NuGraphBlock
from .types import TD

class NuGraphOptical(torch.nn.Module):
    """
    NuGraph optical message-passing engine

    This module incorporates optical information into NuGraph

    Args:
        interaction_features: Number of features in interaction embedding
        nexus_features: Number of features in nexus embedding
        ophit_features: Number of features in optical hit embedding
        pmt_features: Number of features in PMT (flashsumpe) embedding
        flash_features: Number of features in optical flash embedding
        use_checkpointing: Whether to use checkpointing
    """
    def __init__(self, # pylint: disable=too-many-arguments,too-many-positional-arguments
                 interaction_features: int,
                 nexus_features: int,
                 ophit_features: int,
                 pmt_features: int,
                 flash_features: int,
                 use_checkpointing: bool = True):
        super().__init__()

        self.use_checkpointing = use_checkpointing

        # hierarchical message-passing for optical system
        self.ophit_to_pmt = NuGraphBlock(ophit_features, pmt_features, pmt_features)
        self.pmt_to_flash = NuGraphBlock(pmt_features, flash_features, flash_features)
        self.flash_to_interaction = NuGraphBlock(flash_features,
                                                 interaction_features,
                                                 interaction_features)
        self.interaction_to_flash = NuGraphBlock(interaction_features,
                                                 flash_features, flash_features)
        self.flash_to_pmt = NuGraphBlock(flash_features, pmt_features, pmt_features)
        self.pmt_to_ophit = NuGraphBlock(pmt_features, ophit_features, ophit_features)

        # message-passing between PMT nodes
        self.pmt_to_pmt = NuGraphBlock(pmt_features, pmt_features, pmt_features)

        # message-passing between nexus (spacepoint) nodes and PMT nodes
        self.nexus_to_pmt = NuGraphBlock(nexus_features, pmt_features, pmt_features)
        self.pmt_to_nexus = NuGraphBlock(pmt_features, nexus_features, nexus_features)

    def checkpoint(self, net: torch.nn.Module, *args) -> TD:
        """
        Checkpoint module, if enabled.

        Args:
            net: Network module
            args: Arguments to network module
        """
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(net, *args, use_reentrant=False)
        return net(*args)

    def forward(self, data: NuGraphData) -> None:
        """
        NuGraphOptical forward pass

        Args:
            data: Graph data object
        """

        # message-passing from ophit to pmt
        data["pmt"].x = self.checkpoint(
            self.ophit_to_pmt, (data["ophit"].x, data["pmt"].x),
            data["ophit", "in", "pmt"].edge_index)

        # message-passing from pmt to spacepoint
        pmt_sp_edges = data["sp", "knn", "pmt"]
        if "edge_index" in pmt_sp_edges and pmt_sp_edges.edge_index.numel() > 0:
            data["sp"].x = self.checkpoint(
                self.pmt_to_nexus, (data["pmt"].x, data["sp"].x),
                pmt_sp_edges.edge_index[(1,0), :])

        # message-passing from PMTs to PMTs
        pmt_edges = data["pmt", "knn", "pmt"]
        if "edge_index" in pmt_edges and pmt_edges.edge_index.numel() > 0:
            data["pmt"].x = self.checkpoint(
                self.pmt_to_pmt, (data["pmt"].x, data["pmt"].x),
                pmt_edges.edge_index)

        # message-passing from pmt to flash
        data["flash"].x = self.checkpoint(
            self.pmt_to_flash, (data["pmt"].x, data["flash"].x),
            data["pmt", "in", "flash"].edge_index)

        # message-passing from flash to interaction
        data["evt"].x = self.checkpoint(
            self.flash_to_interaction, (data["flash"].x, data["evt"].x),
            data["flash", "in", "evt"].edge_index)

        # message-passing from interaction to flash
        data["flash"].x = self.checkpoint(
            self.interaction_to_flash, (data["evt"].x, data["flash"].x),
            data["flash", "in", "evt"].edge_index[(1,0), :])

        # message-passing from flash to pmt
        data["pmt"].x = self.checkpoint(
            self.flash_to_pmt, (data["flash"].x, data["pmt"].x),
            data["pmt", "in", "flash"].edge_index[(1,0), :])

        # message-passing from PMTs to PMTs
        if "edge_index" in pmt_edges and pmt_edges.edge_index.numel() > 0:
            data["pmt"].x = self.checkpoint(
                self.pmt_to_pmt, (data["pmt"].x, data["pmt"].x),
                pmt_edges.edge_index[(1,0), :])

        # message-passing from spacepoint to pmt
        if "edge_index" in pmt_sp_edges and pmt_sp_edges.edge_index.numel() > 0:
            data["pmt"].x = self.checkpoint(
                self.nexus_to_pmt, (data["sp"].x, data["pmt"].x),
                pmt_sp_edges.edge_index)

        # message-passing from pmt to ophit
        data["ophit"].x = self.checkpoint(
            self.pmt_to_ophit, (data["pmt"].x, data["ophit"].x),
            data["ophit", "in", "pmt"].edge_index[(1,0), :])
