"""NuGraph3 model architecture"""
import argparse
import warnings

import torch
import torch.cuda
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from pytorch_lightning import LightningModule

from .nugraph_types import Data
from .transform import Transform
from .encoder import Encoder
from .core import NuGraphCore
from .decoders import (SemanticDecoder, FilterDecoder, EventDecoder, VertexDecoder, InstanceDecoder,
                       SpacepointDecoder)

from ...data import H5DataModule

#if torch.cuda.is_available():
#    from rmm.allocators.torch import rmm_torch_allocator
#    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

class MichelRegularizer:
    """
    Physics-based Michel electron regularizer targeting 30 MeV per complete Michel electron
    Uses instance decoder clustering to group complete Michel electron tracks
    """
    def __init__(self, lambda_param=0.1, verbose=False):
        self.lambda_param = lambda_param
        self.verbose = verbose

    def __call__(self, batch):
        """Apply Michel physics regularization using instance decoder outputs"""
        try:
            hit_store = batch.get_node_store('hit')
            
            # Use instance decoder output edges
            if ('hit', 'cluster', 'particle') in batch.edge_types:
                return self._instance_based_regularization(batch, ('hit', 'cluster', 'particle'))
            
            # No regularization if no clustering available
            device = hit_store['x_raw'].device
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        except Exception as e:
            if self.verbose:
                print(f"   Regularization error: {e}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def _instance_based_regularization(self, batch, edge_type):
        """Apply 30 MeV regularization to complete Michel electron instances"""
        try:
            hit_store = batch.get_node_store('hit')
        
            # Get hit-to-instance mapping from instance decoder
            edge_index = batch[edge_type].edge_index
            hit_indices = edge_index[0]
            instance_indices = edge_index[1]

            # Check bounds for safety
            max_hit_index = hit_store['x_raw'].size(0) - 1
            valid_mask = hit_indices <= max_hit_index

            if not torch.all(valid_mask):
                if self.verbose:
                    print(f"   WARNING: Found invalid hit indices, filtering them out")
                hit_indices = hit_indices[valid_mask]
                instance_indices = instance_indices[valid_mask]
        
            # Get Michel probabilities from semantic predictions
            probabilities = torch.softmax(hit_store['x_semantic'], dim=1)
            michel_probs = probabilities[:, 3]  # Michel class
        
            # Process each instance
            unique_instances = torch.unique(instance_indices)
            total_reg_loss = torch.tensor(0.0, device=hit_store['x_raw'].device, requires_grad=True)
        
            for instance_id in unique_instances:
                instance_mask = instance_indices == instance_id
                instance_hits = hit_indices[instance_mask]

                if torch.any(instance_hits >= hit_store['x_raw'].size(0)):
                    continue
            
                # Check if this instance is likely a Michel electron
                instance_michel_probs = michel_probs[instance_hits]
                avg_michel_prob = torch.mean(instance_michel_probs)

                if avg_michel_prob > 0.15:  # Michel electron threshold
                    # STEP 1: Sum energy for complete Michel electron instance
                    instance_energies = hit_store['x_raw'][instance_hits, 2]
                    total_energy_raw = torch.sum(instance_energies)
                    
                    # STEP 2: Convert to MeV using Landau conversion
                    total_energy_mev = total_energy_raw * 0.00580717
                
                    # STEP 3: Apply 30 MeV Gaussian penalty for complete Michel electron
                    target_energy = 30.0  # MeV per complete Michel electron
                
                    if total_energy_mev > 5.0:  # Only regularize reasonable energies
                        sigma = 10.0  # Tolerance around 30 MeV
                        prob_weight = avg_michel_prob  # Weight by Michel probability
                    
                        # Gaussian penalty around target
                        energy_diff = total_energy_mev - target_energy
                        gaussian_penalty = torch.exp(-0.5 * (energy_diff / sigma)**2)
                        instance_loss = self.lambda_param * prob_weight * (1 - gaussian_penalty) * 10
                    
                        total_reg_loss = total_reg_loss + instance_loss
        
            return total_reg_loss
        
        except Exception as e:
            if self.verbose:
                print(f"   Instance regularization error: {e}")
            device = hit_store['x_raw'].device
            return torch.tensor(0.0, device=device, requires_grad=True)


class NuGraph3(LightningModule):
    """
    NuGraph3 model architecture.

    Args:
        in_features: Number of input node features
        hit_features: Number of hit node features
        nexus_features: Number of nexus node features
        interaction_features: Number of interaction node features
        instance_features: Number of instance features
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
        use_checkpointing: Whether to use checkpointing
        lr: Learning rate
        enable_michel_reg: Whether to enable Michel physics regularization
        michel_reg_lambda: Strength of Michel regularization
    """
    def __init__(self,
                 in_features: int = 4,
                 hit_features: int = 128,
                 nexus_features: int = 32,
                 interaction_features: int = 32,
                 instance_features: int = 8,
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
                 use_checkpointing: bool = False,
                 lr: float = 0.001,
                 enable_michel_reg: bool = False,
                 michel_reg_lambda: float = 0.1):
        super().__init__()

        warnings.filterwarnings("ignore", ".*NaN values found in confusion matrix.*")

        self.save_hyperparameters()

        self.nexus_features = nexus_features
        self.interaction_features = interaction_features

        self.semantic_classes = semantic_classes
        self.event_classes = event_classes
        self.num_iters = num_iters
        self.lr = lr
        
        # Michel physics regularization
        self.enable_michel_reg = enable_michel_reg
        self.michel_reg_lambda = michel_reg_lambda
        if self.enable_michel_reg:
            self.michel_regularizer = MichelRegularizer(lambda_param=michel_reg_lambda, verbose=False)

        self.encoder = Encoder(in_features, hit_features,
                               nexus_features, interaction_features)

        self.core_net = NuGraphCore(hit_features,
                                    nexus_features,
                                    interaction_features,
                                    use_checkpointing)

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
            self.instance_decoder = InstanceDecoder(hit_features, instance_features)
            self.decoders.append(self.instance_decoder)

        if spacepoint_head:
            self.spacepoint_decoder = SpacepointDecoder(hit_features, len(planes))
            self.decoders.append(self.spacepoint_decoder)

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
        self.encoder(data)
        for _ in range(self.num_iters):
            self.core_net(data)
        total_loss = 0.
        total_metrics = {}
        for decoder in self.decoders:
            loss, metrics = decoder(data, stage)
            total_loss += loss
            total_metrics.update(metrics)

        return total_loss, total_metrics

    def training_step(self,
                      batch: Data,
                      batch_idx: int) -> float:
        loss, metrics = self(batch, 'train')
        
        # Apply Michel physics regularization if enabled
        if self.enable_michel_reg and hasattr(self, 'michel_regularizer'):
            try:
                michel_reg_loss = self.michel_regularizer(batch)
                total_loss = loss + michel_reg_loss
                
                # Log all losses
                self.log('loss/train', loss, batch_size=batch.num_graphs, prog_bar=True)
                self.log('michel_reg_loss', michel_reg_loss, batch_size=batch.num_graphs)
                self.log('total_loss', total_loss, batch_size=batch.num_graphs)
                self.log_dict(metrics, batch_size=batch.num_graphs)
                
                return total_loss
                
            except Exception as e:
                # Fallback to standard loss if regularization fails
                print(f"Michel regularization failed: {e}")
        
        # Standard training without regularization
        self.log('loss/train', loss, batch_size=batch.num_graphs, prog_bar=True)
        self.log_dict(metrics, batch_size=batch.num_graphs)
        return loss

    def on_train_epoch_end(self) -> None:
        # stop updating running average for feature norm
#        self.encoder.input_norm.update = False
        pass

    def validation_step(self,
                        batch,
                        batch_idx: int) -> None:
        loss, metrics = self(batch, 'val')
        self.log('loss/val', loss, batch_size=batch.num_graphs)
        self.log_dict(metrics, batch_size=batch.num_graphs)

    def on_validation_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'val', epoch)

    def test_step(self,
                  batch,
                  batch_idx: int = 0) -> None:
        loss, metrics = self(batch, 'test')
        self.log('loss/test', loss, batch_size=batch.num_graphs)
        self.log_dict(metrics, batch_size=batch.num_graphs)

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
        
        # Use CosineAnnealingLR instead of OneCycleLR to avoid step counting issues
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs * 300000,  # Approximate steps per epoch
            eta_min=self.lr * 0.01  # Minimum learning rate
        )
        
        return [optimizer], {'scheduler': scheduler, 'interval': 'step'}

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
        model.add_argument('--instance-feats', type=int, default=8,
                           help='Hidden dimensionality of object condensation')
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
        model.add_argument('--no-checkpointing', action='store_false',
                           dest="use_checkpointing",
                           help='Disable checkpointing during training')
        model.add_argument('--epochs', type=int, default=80,
                           help='Maximum number of epochs to train for')
        model.add_argument('--learning-rate', type=float, default=0.001,
                           help='Max learning rate during training')
        # Michel physics regularization arguments
        model.add_argument('--enable-michel-reg', action='store_true',
                           help='Enable Michel physics regularization (30 MeV constraint)')
        model.add_argument('--michel-reg-lambda', type=float, default=0.1,
                           help='Strength of Michel regularization')
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
            instance_features=args.instance_feats,
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
            use_checkpointing=args.use_checkpointing,
            lr=args.learning_rate,
            enable_michel_reg=getattr(args, 'enable_michel_reg', False),
            michel_reg_lambda=getattr(args, 'michel_reg_lambda', 0.1))


def main():
    """Main training function when run as script"""
    import pytorch_lightning as pl
    from pathlib import Path
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train NuGraph3 with Michel Physics')
    parser.add_argument('--data-path', type=str, default='/nugraph/NG2-paper.gnn.h5')
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--semantic', action='store_true', help='Enable semantic head')
    parser.add_argument('--filter', action='store_true', help='Enable filter head') 
    parser.add_argument('--instance', action='store_true', help='Enable instance head')
    parser.add_argument('--enable-michel-reg', action='store_true', help='Enable Michel physics')
    parser.add_argument('--michel-reg-lambda', type=float, default=0.1, help='Michel reg strength')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data: {args.data_path}")
    nudata = H5DataModule(args.data_path, batch_size=args.batch_size)
    transform = NuGraph3.transform(nudata.planes)
    nudata.transform = transform
    
    # Create model
    print("Creating NuGraph3 model")
    model = NuGraph3(
        in_features=5,
        hit_features=64,
        nexus_features=16,
        interaction_features=16,
        instance_features=8,
        planes=nudata.planes,
        semantic_classes=nudata.semantic_classes,
        event_classes=nudata.event_classes,
        num_iters=3,
        semantic_head=args.semantic,
        filter_head=args.filter,
        instance_head=args.instance,
        lr=args.learning_rate,
        enable_michel_reg=args.enable_michel_reg,
        michel_reg_lambda=args.michel_reg_lambda
    )
    
    # Setup training
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        save_last=True
    )
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True
    )
    
    # Train
    reg_status = "WITH Michel physics" if args.enable_michel_reg else "WITHOUT Michel physics"
    print(f" Training {reg_status} for {args.epochs} epochs")
    trainer.fit(model, nudata)
    print("Training completed")


if __name__ == '__main__':
    main()
