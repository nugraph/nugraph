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
        ophit_features: Number of features in optical hit embedding
        pmt_features: Number of features in PMT (flashsumpe) embedding
        flash_features: Number of features in optical flash embedding
        use_checkpointing: Whether to use checkpointing
    """
    def __init__(self, # pylint: disable=too-many-arguments,too-many-positional-arguments
                 interaction_features: int,
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
        NuGraphCore forward pass
        
        Args:
            data: Graph data object
        """

        # message-passing from ophit to pmt
        data["opflashsumpe"].x = self.checkpoint(
            self.ophit_to_pmt, (data["ophits"].x, data["opflashsumpe"].x),
            data["ophits", "sumpe", "opflashsumpe"].edge_index)

        # message-passing from pmt to flash
        data["opflash"].x = self.checkpoint(
            self.pmt_to_flash, (data["opflashsumpe"].x, data["opflash"].x),
            data["opflashsumpe", "flash", "opflash"].edge_index)

        # message-passing from flash to interaction
        data["evt"].x = self.checkpoint(
            self.flash_to_interaction, (data["opflash"].x, data["evt"].x),
            data["opflash", "in", "evt"].edge_index)

        # message-passing from interaction to flash
        data["opflash"].x = self.checkpoint(
            self.interaction_to_flash, (data["evt"].x, data["opflash"].x),
            data["opflash", "in", "evt"].edge_index[(1,0), :])

        # message-passing from flash to pmt
        data["opflashsumpe"].x = self.checkpoint(
            self.flash_to_pmt, (data["opflash"].x, data["opflashsumpe"].x),
            data["opflashsumpe", "flash", "opflash"].edge_index[(1,0), :])

        # message-passing from pmt to ophit
        data["ophits"].x = self.checkpoint(
            self.pmt_to_ophit, (data["opflashsumpe"].x, data["ophits"].x),
            data["ophits", "sumpe", "opflashsumpe"].edge_index[(1,0), :])
