"""NuGraph2 network architecture module"""
import argparse
import warnings

import torch
from pytorch_lightning import LightningModule
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import unbatch

from .encoder import Encoder
from .plane import PlaneNet
from .nexus import NexusNet
from .decoders import SemanticDecoder, FilterDecoder
from .transform import Transform

from ...data import H5DataModule

T = torch.Tensor
TD = dict[str, T]

class NuGraph2(LightningModule): # pylint: disable=too-many-instance-attributes
    """
    NuGraph2 model architecture

    Wrap the base model in a LightningModule wrapper to handle training and
    inference, and compute training metrics.
    
    Args:
        in_features: Number of input features
        planar_features: Number of planar features
        nexus_features: Number of nexus features
        planes: Tuple of plane names
        semantic_classes: Tuple of semantic class names
        num_iters: Number of message-passing iterations
        semantic_head: Whether to enable semantic decoder
        filter_head: Whether to enable filter decoder
        checkpoint: Whether to use checkpointing during training
        lr: Maximum learning rate
    """
    def __init__(self, # pylint: disable=too-many-arguments,too-many-positional-arguments
                 in_features: int = 4,
                 planar_features: int = 64,
                 nexus_features: int = 16,
                 planes: tuple[str] = ('u','v','y'),
                 semantic_classes: tuple[str] = ('MIP','HIP','shower','michel','diffuse'),
                 num_iters: int = 5,
                 semantic_head: bool = True,
                 filter_head: bool = True,
                 checkpoint: bool = False,
                 lr: float = 0.001):
        super().__init__()

        warnings.filterwarnings("ignore", ".*NaN values found in confusion matrix.*")

        self.save_hyperparameters()

        self.planes = planes
        self.semantic_classes = semantic_classes
        self.num_iters = num_iters
        self.lr = lr

        self.encoder = Encoder(in_features,
                               planar_features,
                               planes,
                               semantic_classes)

        self.plane_net = PlaneNet(in_features,
                                  planar_features,
                                  len(semantic_classes),
                                  planes,
                                  checkpoint=checkpoint)

        self.nexus_net = NexusNet(planar_features,
                                  nexus_features,
                                  len(semantic_classes),
                                  planes,
                                  checkpoint=checkpoint)

        self.decoders = []

        if semantic_head:
            self.semantic_decoder = SemanticDecoder(
                planar_features,
                planes,
                semantic_classes)
            self.decoders.append(self.semantic_decoder)

        if filter_head:
            self.filter_decoder = FilterDecoder(
                planar_features,
                planes,
                semantic_classes)
            self.decoders.append(self.filter_decoder)

        if not self.decoders:
            raise RuntimeError('At least one decoder head must be enabled!')

    def forward(self, x: TD, edge_index_plane: TD, edge_index_nexus: TD, # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments
                nexus: T, batch: TD) -> TD:
        m = self.encoder(x)
        for _ in range(self.num_iters):
            # shortcut connect features
            for p in self.planes:
                s = x[p].detach().unsqueeze(1).expand(-1, m[p].size(1), -1)
                m[p] = torch.cat((m[p], s), dim=-1)
            self.plane_net(m, edge_index_plane)
            self.nexus_net(m, edge_index_nexus, nexus)
        ret = {}
        for decoder in self.decoders:
            ret.update(decoder(m, batch))
        return ret

    def step(self, data: HeteroData | Batch): # pylint: disable=too-many-branches
        """
        NuGraph2 step function

        Args:
            data: Graph data object
        """
        # if it's a single data instance, convert to batch manually
        if isinstance(data, Batch):
            batch = data
        else:
            batch = Batch.from_data_list([data])

        # unpack tensors to pass into forward function
        x = self(batch.collect('x'),
                 { p: batch[p, 'plane', p].edge_index for p in self.planes },
                 { p: batch[p, 'nexus', 'sp'].edge_index for p in self.planes },
                 torch.empty(batch['sp'].num_nodes, 0),
                 { p: batch[p].batch for p in self.planes })

        # append output tensors back onto input data object
        if isinstance(data, Batch):
            dlist = [ HeteroData() for i in range(data.num_graphs) ]
            for attr, planes in x.items():
                for p, t in planes.items():
                    if t.size(0) == data[p].num_nodes:
                        tlist = unbatch(t, data[p].batch)
                    elif t.size(0) == data.num_graphs:
                        tlist = unbatch(t, torch.arange(data.num_graphs))
                    else:
                        raise RuntimeError(f"Don't know how to unbatch attribute {attr}")
                    for it_d, it_t in zip(dlist, tlist):
                        it_d[p][attr] = it_t
            tmp = Batch.from_data_list(dlist)
            data.update(tmp)
            for attr, planes in x.items():
                for p in planes:
                    # pylint: disable=protected-access
                    data._slice_dict[p][attr] = tmp._slice_dict[p][attr]
                    data._inc_dict[p][attr] = tmp._inc_dict[p][attr]

        else:
            for key, value in x.items():
                data.set_value_dict(key, value)

    def training_step(self, batch) -> float: # pylint: disable=arguments-differ
        self.step(batch)
        total_loss = 0.
        for decoder in self.decoders:
            loss, metrics = decoder.loss(batch, 'train')
            total_loss += loss
            self.log_dict(metrics, batch_size=batch.num_graphs)
        self.log('loss/train', total_loss, batch_size=batch.num_graphs, prog_bar=True)
        return total_loss

    def validation_step(self, batch) -> None: # pylint: disable=arguments-differ
        self.step(batch)
        total_loss = 0.
        for decoder in self.decoders:
            loss, metrics = decoder.loss(batch, 'val')
            total_loss += loss
            self.log_dict(metrics, batch_size=batch.num_graphs)
        self.log('loss/val', total_loss, batch_size=batch.num_graphs)

    def on_validation_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'val', epoch)

    def test_step(self, batch) -> None: # pylint: disable=arguments-differ
        self.step(batch)
        total_loss = 0.
        for decoder in self.decoders:
            loss, metrics = decoder.loss(batch, 'test')
            total_loss += loss
            self.log_dict(metrics, batch_size=batch.num_graphs)
        self.log('loss/test', total_loss, batch_size=batch.num_graphs)

    def on_test_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'test', epoch)

    def predict_step(self, batch: Batch) -> Batch: # pylint: disable=arguments-differ
        self.step(batch)
        return batch

    def configure_optimizers(self) -> tuple:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        onecycle = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], {'scheduler': onecycle, 'interval': 'step'}

    @staticmethod
    def transform(planes: tuple[str]) -> Transform:
        """
        Return data transform for NuGraph2 model
        
        Args:
            planes: tuple of detector plane names
        """
        return Transform(planes)

    @staticmethod
    def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add argparse arguments for model structure
        
        Args:
            parser: Argument parser to append arguments to
        """
        model = parser.add_argument_group('model', 'NuGraph2 model configuration')
        model.add_argument('--num-iters', type=int, default=5,
                           help='Number of message-passing iterations')
        model.add_argument('--in-feats', type=int, default=4,
                           help='Number of input node features')
        model.add_argument('--planar-feats', type=int, default=64,
                           help='Hidden dimensionality of planar convolutions')
        model.add_argument('--nexus-feats', type=int, default=16,
                           help='Hidden dimensionality of nexus convolutions')
        model.add_argument('--semantic', action='store_true', default=False,
                           help='Enable semantic segmentation head')
        model.add_argument('--filter', action='store_true', default=False,
                           help='Enable background filter head')
        model.add_argument('--no-checkpointing', action='store_true', default=False,
                           help='Disable checkpointing during training')
        model.add_argument('--epochs', type=int, default=80,
                           help='Maximum number of epochs to train for')
        model.add_argument('--learning-rate', type=float, default=0.001,
                           help='Max learning rate during training')
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace, nudata: H5DataModule) -> 'NuGraph2':
        """
        Construct NuGraph2 model instance from arguments

        Args:
            args: Argument namespace to initialize model from
            nudata: Input data module
        """
        return cls(
            in_features=args.in_feats,
            planar_features=args.planar_feats,
            nexus_features=args.nexus_feats,
            planes=nudata.planes,
            semantic_classes=nudata.semantic_classes,
            num_iters=args.num_iters,
            semantic_head=args.semantic,
            filter_head=args.filter,
            checkpoint=not args.no_checkpointing,
            lr=args.learning_rate)
