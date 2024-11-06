"""NuGraph core message-passing engine"""
import torch
from torch import nn
from torch_geometric.nn import MessagePassing, HeteroConv
from .types import T, TD

# # Change 4: Single Linear Layer with Mish
# class Mish(nn.Module):
#     def forward(self, x):
#         return x * torch.tanh(nn.functional.softplus(x))

class NuGraphBlock(MessagePassing):
    """
    Standard NuGraph message-passing block
    
    This block generates attention weights for each graph edge based on both
    the source and target node features, and then applies those weights to
    the source node features in order to form messages. These messages are
    then aggregated into the target nodes using softmax aggregation, and
    then fed into a two-layer MLP to generate updated target node features.

    Args:
        source_features: Number of source node input features
        target_features: Number of target node input features
        out_features: Number of target node output features
    """
    def __init__(self, source_features: int, target_features: int,
                 out_features: int):
        super().__init__(aggr="softmax")

        # Original MLP
        self.edge_net = nn.Sequential(
            nn.Linear(source_features+target_features, source_features),
            nn.Sigmoid())
        
        # # Change 1: Single Linear Layer with Sigmoid Activation
        # self.edge_net = nn.Sequential(
        #     nn.Linear(source_features + target_features, 1),
        #     nn.Sigmoid())

        # # Change 2: Single Linear Layer with ReLU
        # self.edge_net = nn.Sequential(
        #     nn.Linear(source_features + target_features, source_features),
        #     nn.ReLU())

        # # Change 3: Single Linear Layer with Leaky ReLU
        # self.edge_net = nn.Sequential(
        #     nn.Linear(source_features + target_features, source_features),
        #     nn.LeakyReLU(negative_slope=0.01))

        # # Change 4: Single Linear Layer with Mish
        # self.edge_net = nn.Sequential(
        #     nn.Linear(source_features + target_features, source_features),
        #     Mish())

        self.net = nn.Sequential(
            nn.Linear(source_features+target_features, out_features),
            nn.Tanh(),
            nn.Linear(out_features, out_features),
            nn.Tanh())

    def forward(self, x: T, edge_index: T) -> T:
        """
        NuGraphBlock forward pass
        
        Args:
            x: Node feature tensor
            edge_index: Edge index tensor
        """
        return self.propagate(edge_index, x=x)

    def message(self, x_i: T, x_j: T) -> T:
        """
        NuGraphBlock message function

        This function constructs messages on graph edges. Features from the
        source and target nodes are concatenated and fed into a linear layer
        to construct attention weights. Messages are then formed on edges by
        weighting the source node features by these attention weights.
        
        Args:
            x_i: Edge features from target nodes
            x_j: Edge features from source nodes
        """
        return self.edge_net(torch.cat((x_i, x_j), dim=1).detach()) * x_j

    def update(self, aggr_out: T, x: T) -> T:
        """
        NuGraphBlock update function

        This function takes the output node features and combines them with
        the input features

        Args:
            aggr_out: Tensor of aggregated node features
            x: Target node features
        """
        if isinstance(x, tuple):
            _, x = x
        return self.net(torch.cat((aggr_out, x), dim=1))

class NuGraphCore(nn.Module):
    """
    NuGraph core message-passing engine
    
    This is the core NuGraph message-passing loop

    Args:
        planar_features: Number of features in planar embedding
        nexus_features: Number of features in nexus embedding
        interaction_features: Number of features in interaction embedding
        ophit_features: Number of features in optical hit embedding
        pmt_features: Number of features in PMT (flashsumpe) embedding
        flash_features: Number of features in optical flash embedding
        planes: List of detector planes
    """
    def __init__(self,
                 planar_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 ophit_features: int,
                 pmt_features: int,
                 flash_features: int,
                 planes: list[str]):
        super().__init__()

        full_nexus_features = len(planes) * nexus_features

        # internal planar message-passing
        self.plane_net = HeteroConv({
            (p, "plane", p): NuGraphBlock(planar_features,
                                          planar_features,
                                          planar_features)
            for p in planes})

        # message-passing from planar nodes to nexus nodes
        self.plane_to_nexus = HeteroConv({
            (p, "nexus", "sp"): NuGraphBlock(planar_features,
                                             nexus_features,
                                             nexus_features)
            for p in planes}, aggr="cat")
        
        # message-passing from optical hit nodes to PMT nodes
        self.hit_to_pmt = HeteroConv({
           ('ophits', 'sumpe', 'opflashsumpe'): NuGraphBlock(ophit_features,
                                                 pmt_features,
                                                 pmt_features)})
        
        # message-passing from PMT nodes to flash nodes
        self.pmt_to_flash = HeteroConv({
           ('opflashsumpe', 'flash', 'opflash'): NuGraphBlock(pmt_features,
                                                flash_features,
                                                flash_features)})

        # message-passing from nexus and flash nodes to interaction nodes
        self.nexus_flash_to_interaction = HeteroConv({
            ("sp", "in", "evt"): NuGraphBlock(full_nexus_features,
                                              interaction_features,
                                              interaction_features), 
            ('opflash', 'in', 'evt'): NuGraphBlock(flash_features, 
                                                interaction_features, 
                                                interaction_features)})

        # message-passing from interaction nodes to nexus nodes
        self.interaction_to_nexus = HeteroConv({
            ("evt", "owns", "sp"): NuGraphBlock(interaction_features,
                                                full_nexus_features,
                                                nexus_features)})

        # message-passing from nexus nodes to planar nodes
        self.nexus_to_plane = HeteroConv({
            ("sp", "nexus", p): NuGraphBlock(nexus_features,
                                             planar_features,
                                             planar_features)
            for p in planes})


        # message-passing from interaction nodes to flash node
        self.interaction_to_flash = HeteroConv({
            ("evt", "in", "opflash"): NuGraphBlock(interaction_features,
                                                   flash_features,
                                                   flash_features)})
        
        # message-passing from flash node to PMT nodes
        self.flash_to_pmt = HeteroConv({
            ("opflash", "flash", "opflashsumpe"): NuGraphBlock(flash_features,
                                                pmt_features,
                                                pmt_features)})

        # message-passing from PMT nodes to optical hit nodes
        self.pmt_to_ophit = HeteroConv({
            ("opflashsumpe", "sumpe", "ophits"): NuGraphBlock(pmt_features,
                                                ophit_features,
                                                ophit_features)})

    def forward(self, p: TD, n: TD,  oph: TD, pmt: TD,
                opf: TD, i: TD, edges: TD) -> tuple[TD, TD, TD, TD, TD, TD]:
        """
        NuGraphCore forward pass
        
        Args:
            p: Planar embedding tensor dictionary
            n: Nexus embedding tensor dictionary
            i: Interaction embedding tensor dictionary
            oph: Optical hit embedding tensor dictionary
            pmt: PMT (flashsumpe) embedding tensor dictionary
            opf: Optical flash embedding tensor dictionary
            edges: Edge index tensor dictionary
        """ 
        # one upward pass
        p = self.plane_net(p, edges)
        n = self.plane_to_nexus(p|n, edges)
        pmt = self.hit_to_pmt(oph|pmt, edges)
        opf = self.pmt_to_flash(pmt|opf, edges)
        i = self.nexus_flash_to_interaction(n|opf|i, edges)

        # one downward pass
        n = self.interaction_to_nexus(i|n, edges)
        p = self.nexus_to_plane(n|p, edges)
        opf = self.interaction_to_flash(i|opf, edges)
        pmt = self.flash_to_pmt(opf|pmt, edges)
        oph = self.pmt_to_ophit(pmt|oph, edges)

        return p, n, oph, pmt, opf, i