import numpy as np
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from argparse import ArgumentParser

# Additional imports for dummy example
from torch_geometric.data import HeteroData

class MichelDistribution:
    @staticmethod
    def get_pdf_value(x, distribution='landau'):
        if distribution == 'landau':
            # Rough Landau approximation (v2.0 style calibration)
            mpv = 50.0   # Most probable value (example)
            eta = 10.0   # Scale parameter (example)
            xi = (x - mpv) / eta
            pdf = np.exp(-0.5 * (xi + np.exp(-xi)))
            return pdf
        elif distribution == 'data':
            # Placeholder for interpolation in a data-driven version.
            return 0.5
        else:
            raise ValueError("Unsupported distribution type.")

class NuGraphModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # Whether Michel energy regularization is enabled.
        self.michelenergy_reg = hparams.get('michelenergy_reg', True)
        # List of semantic class names; note that v2.0 had 5 while v3.0 has 7.
        self.semantic_classes = hparams.get('semantic_classes', ['non-michel', 'michel'])
        self.planes = hparams.get('planes', ['hit', 'sp'])
        # Optionally allow explicit michel_id override.
        self.michel_id = hparams.get('michel_id', None)
        self.reg_type = hparams.get('reg_type', 'cutoff')  # Options: 'cutoff', 'landau', 'data'
        self.michel_reg_cte = hparams.get('michel_reg_cte', 1.0)
        self.lr = hparams.get('learning_rate', 0.001)
        # This flag allows toggling between v2.0 and v3.0 behavior.
        self.simulate_v2 = hparams.get("simulate_v2", False)

    @property
    def michel_class_idx(self):
        """
        Dynamically determine the index corresponding to the Michel class.
        If michel_id is provided explicitly, use that; otherwise find 'michel'.
        """
        if self.michel_id is not None:
            return self.michel_id
        try:
            return self.semantic_classes.index('michel')
        except ValueError:
            return 0  # Default if 'michel' label is not present.

    def step(self, batch, mode='train', is_val=False):
        total_loss = torch.tensor(0.0, device=batch['evt'].x.device)
        total_metrics = {}

        # Use a hyperparameter toggle to switch between v2.0 and v3.0 behavior.
        if self.simulate_v2:
            # ---- v2.0–style Computation (per-hit based) ----
            edep_michel = 0.0
            for p in self.planes:
                # No aggregation: use per-hit features
                y_pred = torch.argmax(batch[p].x_semantic, dim=1)
                michel_idx = self.michel_class_idx
                michel_clusters = (y_pred == michel_idx)
                #print(f"Plane {p}: michel_clusters mask = {michel_clusters}")
                #print(f"Plane {p}: Number of hits classified as Michel = {michel_clusters.sum().item()}")
                # Sum energy integral (assuming index 2) across hits belonging to Michel.
                edep_michel += torch.sum(batch[p].x_raw[michel_clusters, 2])
                #print(f"Plane {p}: edep_michel after summation = {edep_michel}")

            v2_loss = torch.tensor(0.0, device=batch['evt'].x.device)
            edep_lim_high = 160
            edep_lim_low = 1
            # Apply the cutoff penalties (as in v2.0)
            #print(f"Checking thresholds: edep_michel = {edep_michel}, high = {edep_lim_high}, low = {edep_lim_low}")
            if edep_michel > edep_lim_high:
                #print(f"Applying high threshold penalty: edep_michel = {edep_michel} > {edep_lim_high}")
                v2_loss += self.michel_reg_cte * (edep_michel - edep_lim_high) / 15
            if edep_michel < edep_lim_low:
                #print(f"Applying low threshold penalty: edep_michel = {edep_michel} < {edep_lim_low}")
                v2_loss += self.michel_reg_cte * (edep_michel - edep_lim_low) / 10

            total_loss += v2_loss
            total_metrics['edep_michel'] = edep_michel
        else:
            # ---- v3.0–style Computation (cluster-level aggregated features) ----
            michel_reg_loss = 0.0
            edep_michel = 0.0

            edep_lim_high = 160
            edep_lim_low = 1
            pdf_amp = 10

            for p in self.planes:
                # Dynamically determine the number of semantic features:
                num_semantic_features = batch[p].x_semantic.shape[1]
                #print(f"Plane {p}: num_semantic_features = {num_semantic_features}")
                self.log(f'num_semantic_features/{p}', num_semantic_features)

                # Use aggregated cluster-level features: perform argmax on x_semantic.
                y_pred = torch.argmax(batch[p].x_semantic, dim=1)
                michel_idx = self.michel_class_idx
                michel_clusters = (y_pred == michel_idx)
                #print(f"Plane {p}: michel_clusters mask = {michel_clusters}")
                #print(f"Plane {p}: Number of clusters classified as Michel = {michel_clusters.sum().item()}")
                # Aggregate energy from cluster-level x_raw; note scaling by batch.num_graphs.
                sumintegral_michel = torch.sum(batch[p].x_raw[michel_clusters, 2])
                edep_michel += sumintegral_michel * (0.00580717 / batch.num_graphs)
                #print(f"Plane {p}: edep_michel after aggregation = {edep_michel}")

                if edep_michel > 0:
                    if self.reg_type == 'cutoff':
                        if edep_michel > edep_lim_high:
                            #print(f"Applying high threshold penalty: edep_michel = {edep_michel} > {edep_lim_high}")
                            michel_reg_loss += self.michel_reg_cte * (edep_michel - edep_lim_high) / 15
                        if edep_michel < edep_lim_low:
                            #print(f"Applying low threshold penalty: edep_michel = {edep_michel} < {edep_lim_low}")
                            michel_reg_loss += self.michel_reg_cte * (edep_michel - edep_lim_low) / 10
                    elif self.reg_type == 'landau' and edep_michel > 8.5:
                        pdf_value = MichelDistribution.get_pdf_value(edep_michel, distribution='landau')
                        michel_reg_loss += self.michel_reg_cte * (1 - pdf_value) * pdf_amp
                    elif self.reg_type == 'data':
                        pdf_value = MichelDistribution.get_pdf_value(edep_michel, distribution='data')
                        michel_reg_loss += self.michel_reg_cte * (1 - pdf_value) * pdf_amp

            total_loss += michel_reg_loss
            total_metrics['edep_michel'] = edep_michel

        return total_loss, total_metrics

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, metrics = self.step(batch, mode='train')
        #print(f"Training step: Loss = {loss.item()}")
        self.log('loss/train', loss, batch_size=batch.num_graphs, prog_bar=True)
        self.log_dict(metrics, batch_size=batch.num_graphs)
        self.log_memory(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        loss, metrics = self.step(batch, mode='val', is_val=True)
        self.log('loss/val', loss, batch_size=batch.num_graphs)
        self.log_dict(metrics, batch_size=batch.num_graphs)

    def test_step(self, batch, batch_idx: int = 0) -> None:
        loss, metrics = self.step(batch, mode='test', is_val=True)
        self.log('loss/test', loss, batch_size=batch.num_graphs)
        self.log_dict(metrics, batch_size=batch.num_graphs)
        self.log_memory(batch, 'test')

    def predict_step(self, batch, batch_idx: int = 0):
        self.step(batch)
        return batch

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        onecycle = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], {'scheduler': onecycle, 'interval': 'step'}

    def log_memory(self, batch, stage: str) -> None:
        if not hasattr(self, 'max_mem_cpu'):
            self.max_mem_cpu = 0.0
        if self.device != torch.device('cpu'):
            gpu_mem = torch.cuda.memory_reserved(self.device)
            gpu_mem = float(gpu_mem) / 1073741824  # Convert bytes to GB.
            if not hasattr(self, 'max_mem_gpu'):
                self.max_mem_gpu = gpu_mem
            else:
                self.max_mem_gpu = max(self.max_mem_gpu, gpu_mem)
            self.log(f'memory_gpu/{stage}', self.max_mem_gpu,
                     batch_size=batch.num_graphs, reduce_fx=torch.max)

    @staticmethod
    def add_model_args(parser: ArgumentParser) -> ArgumentParser:
        model = parser.add_argument_group('model', 'NuGraph3 model configuration')
        model.add_argument('--planar-feats', type=int, default=128,
                           help='Hidden dimensionality of planar convolutions')
        model.add_argument('--nexus-feats', type=int, default=32,
                           help='Hidden dimensionality of nexus convolutions')
        model.add_argument('--vertex-aggr', type=str, default='lstm',
                           help='Aggregation function for vertex aggregator')
        model.add_argument('--vertex-lstm-feats', type=int, default=32,
                           help='Hidden dimensionality of vertex LSTM aggregation')
        model.add_argument('--vertex-mlp-feats', type=int, nargs='*', default=[32],
                           help='Hidden dimensionality of vertex decoder')
        model.add_argument('--event', action='store_true', default=False,
                           help='Enable event classification head')
        model.add_argument('--semantic', action='store_true', default=False,
                           help='Enable semantic segmentation head')
        model.add_argument('--filter', action='store_true', default=False,
                           help='Enable background filter head')
        model.add_argument('--vertex', action='store_true', default=False,
                           help='Enable vertex regression head')
        return parser

    @staticmethod
    def add_train_args(parser: ArgumentParser) -> ArgumentParser:
        train = parser.add_argument_group('train', 'NuGraph3 training configuration')
        train.add_argument('--no-checkpointing', action='store_true', default=False,
                           help='Disable checkpointing during training')
        train.add_argument('--epochs', type=int, default=80,
                           help='Maximum number of epochs to train for')
        train.add_argument('--learning-rate', type=float, default=0.001,
                           help='Max learning rate during training')
        train.add_argument('--clip-gradients', type=float, default=None,
                           help='Maximum value to clip gradient norm')
        train.add_argument('--gamma', type=float, default=2,
                           help='Focal loss gamma parameter')
        return parser

if __name__ == '__main__':
    # Create a dummy batch for testing purposes.
    batch = HeteroData()
    num_clusters = 10  # Number of clusters (or hits) per plane
    num_classes = 3    # Number of semantic classes (this will show dynamic behavior)

    # Dummy "hit" node storage.
    batch['hit'] = type('DummyNode', (), {})()
    batch['hit'].x_semantic = torch.randn(num_clusters, num_classes)  # Initialize with random values
    michel_idx = 1  # Assuming 'michel' class is at index 1 in semantic_classes
    batch['hit'].x_semantic[:, michel_idx] = torch.tensor([0.99, 0.9, 0.1, 0.85, 0.95, 0.4, 0.2, 0.97, 0.3, 0.88])  # Explicit probabilities for Michel predictions
    
    batch['hit'].x_raw = torch.randn(num_clusters, 5)  # Initialize full shape
    batch['hit'].x_raw[:, 2] = torch.tensor([200, 0.2, 50, 180, 170, 0.8, 90, 190, 0.1, 0.05])  # Energy values at index 2

    # Dummy "sp" node storage.
    batch['sp'] = type('DummyNode', (), {})()
    batch['sp'].x_semantic = torch.randn(num_clusters, num_classes)
    batch['sp'].x_raw = torch.randn(num_clusters, 5)

    # Dummy event node storage to infer device.
    batch['evt'] = type('DummyNode', (), {})()
    batch['evt'].x = torch.randn(1, 5)
    batch['evt'].y = torch.tensor([1])
    batch.num_graphs = 2

    #print(f'x_raw (energy values): {batch["hit"].x_raw[:, 2]}')
    #print(f'x_semantic (Michel probabilities): {batch["hit"].x_semantic[:, michel_idx]}')



    # Dummy hyperparameters dictionary. To test v2.0 behavior, set simulate_v2 to True.
    hparams = {
        'michelenergy_reg': True,
        'semantic_classes': ['non-michel', 'michel'],
        'planes': ['hit', 'sp'],
        'michel_id': None,   # Let the model determine the Michel class index dynamically.
        'reg_type': 'cutoff',  # Options: 'cutoff', 'landau', 'data', we are using cutooff following v2.0 
        'michel_reg_cte': 1.0,
        'learning_rate': 0.001,
        'simulate_v2': False  # Change to True to simulate v2.0 behavior.
    }

    model = NuGraphModel(hparams)
    loss = model.training_step(batch, 0)
    print("Loss:", loss.item())


