"""NuGraph3 model architecture"""
import argparse
import warnings

import torch.cuda
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from pytorch_lightning import LightningModule

from .types import Data
from .transform import Transform
from .encoder import TPCEncoder
from .core import NuGraphCore
from .optical import OpticalEncoder, NuGraphOptical
from .feedback import SemanticFeedback
from .decoders import (SemanticDecoder, FilterDecoder, EventDecoder, VertexDecoder, InstanceDecoder,
                       SpacepointDecoder, NexusDecoder)

from ...data import H5DataModule

class NuGraph3(LightningModule):
    """
    NuGraph3 model architecture.

    Args:
        in_features: Number of input node features
        hit_features: Number of hit node features
        nexus_features: Number of nexus node features
        interaction_features: Number of interaction node features
        beta_features: Number of object condensation beta features
        coord_features: Number of object condensation coordinate features
        instance_features: Number of instance features
        ophit_features: Number of features in optical hit embedding
        pmt_features: Number of features in PMT (flashsumpe) embedding
        flash_features: Number of features in optical flash embedding
        planes: Tuple of detector plane names
        semantic_classes: Tuple of semantic classes
        event_classes: Tuple of event classes
        num_iters: Number of message-passing iterations
        event_head: Whether to enable event decoder
        semantic_head: Whether to enable semantic decoder
        filter_head: Whether to enable filter decoder
        vertex_head: Whether to enable vertex decoder
        instance_head: Whether to enable instance decoder
        spacepoint_head: Whether to enable spacepoint decoder
        nexus_head: Whether to enable nexus decoder and spacepoint graph message-passing
        nexus_in_features: Number of nexus node input features
        nexus_geometry: Whether spacepoint messages see spacepoint displacements
        nexus_vertex: Vertex given to the nexus edge classifier ("pred", "vote" or "true")
        nexus_direction: Whether to regress spacepoint directions for the nexus edge classifier
        nexus_vertex_refine: Whether to fit the vertex from the nexus votes and directions and
            refine it with attention over nearby spacepoints
        nexus_charge: Whether the nexus edge classifier sees hit charge and width across edges
        nexus_knn_shape: Whether the nexus direction head sees k-nearest-neighbour shapes
        nexus_edge_pos_weight: Weight of same-particle edges in the nexus edge loss
        nexus_semantic: Whether the nexus edge classifier sees semantic predictions at both ends
        nexus_balance_filter: Whether to weight the nexus ghost-filter loss by semantic class
        nexus_log_length: Whether the nexus edge classifier sees each edge's log length
        nexus_stub_count: Whether the nexus edge classifier sees the Hough stub counts at both ends
        nexus_edge_layers: Number of hidden layers in the nexus edge classifier
        nexus_edge_width: Width of the nexus edge classifier's hidden layers (0 for nexus_features)
        nexus_feedback: Whether the spacepoint stage feeds back to hits before a second semantic pass
        use_optical: Whether to perform message-passing in optical system
        dbscan_eps: Epsilon hyperparameter for DBSCAN algorithm
        particle_loss: Whether to include particle loss term for object condensation"
        use_checkpointing: Whether to use checkpointing
        residual: Whether core message-passing blocks add residual updates
        lr: Learning rate
        no_one_cycle_sched: Whether to disable the OneCycleLR scheduler
    """
    def __init__(self,
                 in_features: int = 4,
                 hit_features: int = 128,
                 nexus_features: int = 32,
                 interaction_features: int = 32,
                 beta_features: int = 32,
                 coord_features: int = 128,
                 instance_features: int = 8,
                 ophit_features: int = 128,
                 pmt_features: int = 64,
                 flash_features: int = 32,
                 planes: tuple[str] = ("u","v","y"),
                 semantic_classes: tuple[str] = ('MIP','HIP','shower','michel','diffuse'),
                 event_classes: tuple[str] = ('numu','nue','nc'),
                 num_iters: int = 5,
                 event_head: bool = False,
                 semantic_head: bool = True,
                 filter_head: bool = True,
                 vertex_head: bool = False,
                 instance_head: bool = False,
                 spacepoint_head: bool = False,
                 nexus_head: bool = False,
                 nexus_in_features: int = 0,
                 nexus_geometry: bool = False,
                 nexus_vertex: str = None,
                 nexus_direction: bool = False,
                 nexus_vertex_refine: bool = False,
                 nexus_charge: bool = False,
                 nexus_knn_shape: bool = False,
                 nexus_edge_pos_weight: float = 1.,
                 nexus_semantic: bool = False,
                 nexus_balance_filter: bool = False,
                 nexus_log_length: bool = False,
                 nexus_stub_count: bool = False,
                 nexus_edge_layers: int = 1,
                 nexus_edge_width: int = 0,
                 nexus_feedback: bool = False,
                 dbscan_eps: float = 0.3,
                 particle_loss: bool = False,
                 use_optical: bool = False,
                 use_checkpointing: bool = False,
                 residual: bool = False,
                 lr: float = 0.001,
                 no_one_cycle_sched: bool = False):
        super().__init__()

        warnings.filterwarnings("ignore", ".*NaN values found in confusion matrix.*")

        # batches skipped because they did not fit in GPU memory
        self.oom_skipped = 0
        self.oom_in_a_row = 0
        self.oom_scale = None

        self.save_hyperparameters()

        self.nexus_features = nexus_features
        self.interaction_features = interaction_features

        self.semantic_classes = semantic_classes
        self.event_classes = event_classes
        self.num_iters = num_iters
        self.lr = lr
        self.no_one_cycle_sched = no_one_cycle_sched
        self.use_optical = use_optical

        self.tpc_encoder = TPCEncoder(in_features, hit_features,
                                      nexus_features, interaction_features,
                                      beta_features, coord_features,
                                      nexus_in_features)

        self.core_net = NuGraphCore(hit_features,
                                    nexus_features,
                                    interaction_features,
                                    beta_features,
                                    coord_features,
                                    use_checkpointing,
                                    nexus_head,
                                    nexus_geometry,
                                    residual)

        if self.use_optical:
            self.optical_encoder = OpticalEncoder(ophit_features=ophit_features,
                                                  pmt_features=pmt_features,
                                                  flash_features=flash_features)
            self.optical_net = NuGraphOptical(interaction_features=interaction_features,
                                              nexus_features=nexus_features,
                                              ophit_features=ophit_features,
                                              pmt_features=pmt_features,
                                              flash_features=flash_features,
                                              use_checkpointing=use_checkpointing)

        self.decoders = []

        if event_head:
            self.event_decoder = EventDecoder(interaction_features, event_classes)
            self.decoders.append(self.event_decoder)

        if semantic_head:
            self.semantic_decoder = SemanticDecoder(hit_features, semantic_classes)
            self.decoders.append(self.semantic_decoder)

        if filter_head:
            self.filter_decoder = FilterDecoder(hit_features,)
            self.decoders.append(self.filter_decoder)

        if vertex_head:
            self.vertex_decoder = VertexDecoder(interaction_features)
            self.decoders.append(self.vertex_decoder)

        if instance_head:
            self.instance_decoder = InstanceDecoder(beta_features, coord_features,
                                                    instance_features, semantic_classes,
                                                    dbscan_eps, particle_loss)
            self.decoders.append(self.instance_decoder)

        if spacepoint_head:
            self.spacepoint_decoder = SpacepointDecoder(hit_features, len(planes))
            self.decoders.append(self.spacepoint_decoder)

        if nexus_vertex == "pred" and not vertex_head:
            raise RuntimeError("nexus_vertex=\"pred\" requires the vertex head!")
        if nexus_semantic and not semantic_head:
            raise RuntimeError("nexus_semantic requires the semantic head!")
        if nexus_head:
            self.nexus_decoder = NexusDecoder(nexus_features, nexus_vertex, nexus_direction,
                                              interaction_features,
                                              nexus_charge, nexus_knn_shape, nexus_edge_pos_weight,
                                              len(semantic_classes) if nexus_semantic else 0,
                                              nexus_balance_filter, nexus_log_length, nexus_stub_count,
                                              nexus_edge_layers, nexus_edge_width,
                                              vertex_refine=nexus_vertex_refine)
            self.decoders.append(self.nexus_decoder)

        self.feedback = None
        if nexus_feedback:
            if not (semantic_head and nexus_head):
                raise RuntimeError("nexus_feedback requires the semantic and nexus heads!")
            self.feedback = SemanticFeedback(hit_features, nexus_features, len(semantic_classes))

        if not self.decoders:
            raise RuntimeError('At least one decoder head must be enabled!')

    def forward(self, data: Data, stage: str = None): # pylint: disable=arguments-differ
        """
        NuGraph3 forward function

        This function runs the forward pass of the NuGraph3 architecture,
        and then loops over each decoder to compute the loss and calculate
        and log any performance metrics.

        Args:
            data: Graph data object
            stage: String tag defining the step type
        """
        self.tpc_encoder(data)
        if self.use_optical:
            self.optical_encoder(data)
        for _ in range(self.num_iters):
            self.core_net(data)
            if self.use_optical:
                self.optical_net(data)
        total_loss = 0.
        total_metrics = {}
        for decoder in self.decoders:
            # with feedback, the first semantic pass is not reported
            first_pass = self.feedback is not None and decoder is self.semantic_decoder
            loss, metrics = decoder(data, None if first_pass else stage)
            total_loss += loss
            total_metrics.update(metrics)
        if self.feedback is not None:
            self.feedback(data)
            loss, metrics = self.semantic_decoder(data, stage)
            total_loss += loss
            total_metrics.update(metrics)

        return total_loss, total_metrics

    def training_step(self,
                      batch: Data,
                      batch_idx: int) -> float:
        # metrics make the CPU wait on the GPU, so only compute them on steps the logger keeps
        n = self.trainer.log_every_n_steps
        log_step = n > 0 and (self.global_step + 1) % n == 0
        if log_step and self.oom_skipped:
            self.log('oom/skipped', float(self.oom_skipped), batch_size=1)
        if batch is None:  # its graph did not fit in GPU memory
            return self.skip_batch('train', batch_idx)
        oom = False
        try:
            loss, metrics = self(batch, 'train' if log_step else None)
        except torch.OutOfMemoryError:
            oom = True
        if oom:  # outside the except block, so that the failed step's tensors are freed
            return self.skip_batch('train', batch_idx)
        self.log('loss/train', loss, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True)
        self.log_dict(metrics, batch_size=batch.num_graphs, sync_dist=True)
        return loss

    def backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        oom = False
        try:
            super().backward(loss, *args, **kwargs)
        except torch.OutOfMemoryError:
            oom = True
        if oom:
            # without gradients the optimizer step leaves every parameter unchanged
            self.zero_grad(set_to_none=True)
            scaler = getattr(self.trainer.precision_plugin, 'scaler', None)
            if scaler is not None:
                # a gradient scaler refuses to step without gradients, so give one
                # parameter an infinite gradient and let the scaler skip the step
                p = next(p for p in self.parameters() if p.requires_grad)
                p.grad = torch.full_like(p, float('inf'))
                self.oom_scale = scaler.get_scale()
            self.skip_batch('train', None)
        else:
            self.oom_in_a_row = 0

    def skip_batch(self, stage: str, batch_idx: int | None) -> None:
        """
        Skip a batch that did not fit in GPU memory instead of ending the run;
        ten in a row still end it, as does any on more than one GPU, where the
        other processes would wait for this one

        Args:
            stage: Stage name (train/val)
            batch_idx: Index of the batch, if known
        """
        self.oom_skipped += 1
        self.oom_in_a_row += 1
        if self.trainer.world_size > 1 or self.oom_in_a_row >= 10:
            raise torch.OutOfMemoryError(f"{self.oom_in_a_row} batches in a row did not fit in GPU memory")
        torch.cuda.empty_cache()
        warnings.warn(f"{stage} batch {'' if batch_idx is None else batch_idx} skipped: "
                      f"CUDA out of memory ({self.oom_skipped} skipped so far)")

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.oom_scale is not None:  # undo the scaler's backoff for a skipped step
            self.trainer.precision_plugin.scaler.update(self.oom_scale)
            self.oom_scale = None

    def on_train_epoch_end(self) -> None:
        # stop updating running average for feature norm
        self.tpc_encoder.input_norm.update = False
        if self.tpc_encoder.nexus_in_features:
            self.tpc_encoder.nexus_norm.update = False

    def validation_step(self,
                        batch,
                        batch_idx: int) -> None:
        if batch is None:
            self.skip_batch('val', batch_idx)
            return
        oom = False
        try:
            loss, metrics = self(batch, 'val')
        except torch.OutOfMemoryError:
            oom = True
        if oom:
            self.skip_batch('val', batch_idx)
            return
        self.oom_in_a_row = 0
        self.log('loss/val', loss, batch_size=batch.num_graphs, sync_dist=True)
        self.log_dict(metrics, batch_size=batch.num_graphs, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'val', epoch)

    def test_step(self,
                  batch,
                  batch_idx: int = 0) -> None:
        loss, metrics = self(batch, 'test')
        self.log('loss/test', loss, batch_size=batch.num_graphs, sync_dist=True)
        self.log_dict(metrics, batch_size=batch.num_graphs, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'test', epoch)

    def predict_step(self,
                     batch: Data,
                     batch_idx: int = 0) -> Data:
        self(batch)
        return batch

    def configure_optimizers(self) -> tuple:
        optimizer = AdamW(self.parameters(),
                          lr=self.lr)
        if self.no_one_cycle_sched:
            return optimizer
        else:
            onecycle = OneCycleLR(
                       optimizer,
                       max_lr=self.lr,
                       total_steps=self.trainer.estimated_stepping_batches)
            return [optimizer], {'scheduler': onecycle, 'interval': 'step'}

    @staticmethod
    def transform(planes: tuple[str]) -> Transform:
        """
        Return data transform for NuGraph3 model

        Args:
            planes: tuple of detector plane names
        """
        return Transform(planes)

    @staticmethod
    def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add argparse argument group for NuGraph3 model

        Args:
            parser: Argument parser to append argument group to
        """
        model = parser.add_argument_group('model', 'NuGraph3 model configuration')
        model.add_argument('--num-iters', type=int, default=5,
                           help='Number of message-passing iterations')
        model.add_argument('--in-feats', type=int, default=5,
                           help='Number of input node features')
        model.add_argument('--hit-feats', type=int, default=128,
                           help='Hidden dimensionality of hit convolutions')
        model.add_argument('--nexus-feats', type=int, default=32,
                           help='Hidden dimensionality of nexus convolutions')
        model.add_argument('--interaction-feats', type=int, default=32,
                           help='Hidden dimensionality of interaction layer')
        model.add_argument('--beta-feats', type=int, default=32,
                           help='Hidden dimensionality of clustering beta embedding')
        model.add_argument('--coord-feats', type=int, default=128,
                           help='Hidden dimensionality of clustering coordinate embedding')
        model.add_argument('--instance-feats', type=int, default=8,
                           help='Size of clustering embedding')
        model.add_argument('--ophit-features', type=int, default=128,
                           help='Number of optical hit features')
        model.add_argument('--pmt-features', type=int, default=64,
                           help='Number of PMT features')
        model.add_argument('--flash-features', type=int, default=32,
                           help='Number of optical flashes features')
        model.add_argument('--event', action='store_true',
                           help='Enable event classification head')
        model.add_argument('--semantic', action='store_true',
                           help='Enable semantic segmentation head')
        model.add_argument('--filter', action='store_true',
                           help='Enable background filter head')
        model.add_argument('--instance', action='store_true',
                           help='Enable instance segmentation head')
        model.add_argument('--vertex', action='store_true',
                           help='Enable vertex regression head')
        model.add_argument("--spacepoint", action="store_true",
                           help="Enable spacepoint prediction head")
        model.add_argument("--nexus", action="store_true",
                           help="Enable nexus decoder and spacepoint graph message-passing")
        model.add_argument("--nexus-in-feats", type=int, default=0,
                           help="Number of nexus node input features")
        model.add_argument("--nexus-geometry", action="store_true",
                           help="Pass spacepoint displacements into spacepoint messages")
        model.add_argument("--nexus-vertex", type=str, default=None, choices=("pred", "vote", "true"),
                           help="Give the nexus edge classifier edge geometry relative to the "
                                "predicted, spacepoint-voted or (diagnostic only) true vertex")
        model.add_argument("--nexus-direction", action="store_true",
                           help="Regress spacepoint directions for the nexus edge classifier")
        model.add_argument("--nexus-vertex-refine", action="store_true",
                           help="Fit the vertex from the nexus votes and directions and refine it with "
                                "attention over nearby spacepoints (needs --nexus-vertex vote "
                                "--nexus-direction)")
        model.add_argument("--nexus-charge", action="store_true",
                           help="Give the nexus edge classifier hit charge and width across edges "
                                "(needs --sp-charge)")
        model.add_argument("--nexus-knn-shape", action="store_true",
                           help="Give the nexus direction head k-nearest-neighbour neighbourhood shapes")
        model.add_argument("--nexus-edge-pos-weight", type=float, default=1.,
                           help="Weight of same-particle edges in the nexus edge loss")
        model.add_argument("--nexus-semantic", action="store_true",
                           help="Give the nexus edge classifier semantic predictions at both ends")
        model.add_argument("--nexus-balance-filter", action="store_true",
                           help="Weight the nexus ghost-filter loss by true semantic class frequency")
        model.add_argument("--nexus-log-length", action="store_true",
                           help="Give the nexus edge classifier each edge's log length")
        model.add_argument("--nexus-stub-count", action="store_true",
                           help="Give the nexus edge classifier the Hough stub counts at both ends "
                                "(needs --sp-graph gabriel+stubs)")
        model.add_argument("--nexus-edge-layers", type=int, default=1,
                           help="Number of hidden layers in the nexus edge classifier")
        model.add_argument("--nexus-edge-width", type=int, default=0,
                           help="Width of the nexus edge classifier's hidden layers (0 for --nexus-feats)")
        model.add_argument("--nexus-feedback", action="store_true",
                           help="Feed the spacepoint stage back to hits, weighted by edge and ghost "
                                "scores, before a second semantic pass")
        model.add_argument("--dbscan-eps", type=float, default=0.3,
                           help="Epsilon hyperparameter for DBSCAN algorithm")
        model.add_argument("--particle-loss", action="store_true",
                           help="Enable object condensation particle loss term")
        model.add_argument('--optical', action='store_true',
                           help='Enable optical hierarchy')
        model.add_argument('--no-checkpointing', action='store_false',
                           dest="use_checkpointing",
                           help='Disable checkpointing during training')
        model.add_argument('--no-residual', action='store_false',
                           dest="residual",
                           help='Disable residual connections in core message-passing blocks')
        model.add_argument('--epochs', type=int, default=80,
                           help='Maximum number of epochs to train for')
        model.add_argument('--learning-rate', type=float, default=0.001,
                           help='Max learning rate during training')
        model.add_argument('--no-lr-scheduler', action='store_true',
                           dest="no_one_cycle_sched",
                           help='Disable OneCycleLR scheduler')
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace, nudata: H5DataModule) -> 'NuGraph3':
        """
        Construct model from arguments

        Args:
            args: Namespace containing parsed arguments
            nudata: Data module
        """
        return cls(
            in_features=args.in_feats,
            hit_features=args.hit_feats,
            nexus_features=args.nexus_feats,
            interaction_features=args.interaction_feats,
            beta_features=args.beta_feats,
            coord_features=args.coord_feats,
            instance_features=args.instance_feats,
            ophit_features=args.ophit_features,
            pmt_features=args.pmt_features,
            flash_features=args.flash_features,
            planes=nudata.planes,
            semantic_classes=nudata.semantic_classes,
            event_classes=nudata.event_classes,
            num_iters=args.num_iters,
            event_head=args.event,
            semantic_head=args.semantic,
            filter_head=args.filter,
            vertex_head=args.vertex,
            instance_head=args.instance,
            spacepoint_head=args.spacepoint,
            nexus_head=args.nexus,
            nexus_in_features=args.nexus_in_feats,
            nexus_geometry=args.nexus_geometry,
            nexus_vertex=args.nexus_vertex,
            nexus_direction=args.nexus_direction,
            nexus_vertex_refine=args.nexus_vertex_refine,
            nexus_charge=args.nexus_charge,
            nexus_knn_shape=args.nexus_knn_shape,
            nexus_edge_pos_weight=args.nexus_edge_pos_weight,
            nexus_semantic=args.nexus_semantic,
            nexus_balance_filter=args.nexus_balance_filter,
            nexus_log_length=args.nexus_log_length,
            nexus_stub_count=args.nexus_stub_count,
            nexus_edge_layers=args.nexus_edge_layers,
            nexus_edge_width=args.nexus_edge_width,
            nexus_feedback=args.nexus_feedback,
            dbscan_eps=args.dbscan_eps,
            particle_loss=args.particle_loss,
            use_optical=args.optical,
            use_checkpointing=args.use_checkpointing,
            residual=args.residual,
            lr=args.learning_rate,
            no_one_cycle_sched=args.no_one_cycle_sched)
