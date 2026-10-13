"""NuGraph3 encoder"""
import torch
from pynuml.data import NuGraphData
from ...util import InputNorm

class TPCEncoder(torch.nn.Module):
    """
    NuGraph3 TPC encoder

    Args:
        in_features: Number of input node features
        planar_features: Number of planar node features
        nexus_feature: Number of nexus node features
        beta_features: Number of object condensation beta features
        coord_features: Number of object condensation coordinate features
        nexus_in_features: Number of nexus node input features
        plane_geometry: Per plane (readout view), the two axes its hits measure position along
            (wire coordinate and drift in a TPC, cell and plane position in a segmented
            detector), each as five numbers: its 3D direction, the coordinate's value at the
            detector origin and its resolution, cm. If given, the hit's plane index becomes the
            two directions and resolutions, and both coordinates are measured from the origin,
            so shared weights see the same geometry in every detector
        geometry_columns: Input feature columns holding the two coordinates and the plane index
    """
    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 beta_features: int,
                 coord_features: int,
                 nexus_in_features: int = 0,
                 plane_geometry: list[list[float]] = None,
                 geometry_columns: tuple[int] = (0, 1, -1)):
        super().__init__()

        self.register_buffer("plane_geometry", torch.tensor(plane_geometry) if plane_geometry else None,
                             persistent=False)
        if plane_geometry:
            in_features += 7  # plane index -> two axis directions and resolutions
        self.geometry_columns = geometry_columns

        self.input_norm = InputNorm(in_features)
        self.planar_net = torch.nn.Linear(in_features, planar_features)

        # object condensation beta encoder
        self.beta_net = torch.nn.Sequential(
            torch.nn.Linear(in_features, beta_features),
            torch.nn.Mish(),
        )

        # object condensation coordinate encoder
        self.coord_net = torch.nn.Sequential(
            torch.nn.Linear(in_features, coord_features),
            torch.nn.Mish(),
        )

        # nexus input encoder
        if nexus_in_features:
            self.nexus_norm = InputNorm(nexus_in_features)
            self.nexus_net = torch.nn.Linear(nexus_in_features, nexus_features)

        self.nexus_features = nexus_features
        self.nexus_in_features = nexus_in_features
        self.interaction_features = interaction_features

    def forward(self, data: NuGraphData) -> None:
        """
        NuGraph3 TPC encoder forward pass

        Args:
            data: Graph data object
        """
        x_in = data["hit"].x
        if self.plane_geometry is not None:
            g = self.plane_geometry[data["hit"].plane]  # per axis: direction (3), origin, resolution
            first, second, plane = (c % x_in.size(1) for c in self.geometry_columns)
            col = torch.arange(x_in.size(1), device=x_in.device)
            x_in = x_in - g[:, 3:4] * (col == first) - g[:, 8:9] * (col == second)
            x_in = torch.cat((x_in[:, :plane], x_in[:, plane + 1:], g[:, [0, 1, 2, 4, 5, 6, 7, 9]]), dim=1)
        x_in = self.input_norm(x_in)
        data["hit"].x = self.planar_net(x_in)
        data["hit"].of = self.beta_net(x_in)
        data["hit"].ox = self.coord_net(x_in)
        if self.nexus_in_features:
            data["sp"].x = self.nexus_net(self.nexus_norm(data["sp"].x))
        else:
            data["sp"].x = torch.zeros(data["sp"].num_nodes,
                                       self.nexus_features,
                                       device=data["hit"].x.device)
        data["evt"].x = torch.zeros(data["evt"].num_nodes,
                                    self.interaction_features,
                                    device=data["hit"].x.device)
