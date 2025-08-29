#!/usr/bin/env python
"""
Group Michel hits → Sum energies → Apply 30 MeV Gaussian penalty
"""

import os
import sys
import argparse
import torch
import pytorch_lightning as pl
import json
import csv
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import gc
from pathlib import Path

# environment variables
os.environ['WANDB_MODE'] = 'disabled'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

def setup_nugraph_environment():
    """nugraph environment variables"""
    try:
        current_file = os.path.abspath(__file__)
        nugraph_dir = os.path.dirname(current_file)
        
        if nugraph_dir not in sys.path:
            sys.path.insert(0, nugraph_dir)
        
        if 'NUGRAPH_DIR' not in os.environ:
            os.environ['NUGRAPH_DIR'] = nugraph_dir
        if 'NUGRAPH_DATA' not in os.environ:
            os.environ['NUGRAPH_DATA'] = '/nugraph'
        if 'NUGRAPH_LOG' not in os.environ:
            os.environ['NUGRAPH_LOG'] = os.path.join(nugraph_dir, 'logs')
            
        print(f"NuGraph environment setup: {nugraph_dir}")
        return True
        
    except Exception as e:
        print(f"NuGraph environment setup failed: {e}")
        return False

# environment
setup_nugraph_environment()

# matplotlib
plt.switch_backend('Agg')

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Michel regularizer with instance decoder
class MichelRegularizer:
    def __init__(self, lambda_param=0.1, verbose=False):
        self.lambda_param = lambda_param
        self.verbose = verbose

    def __call__(self, batch):
        """instance decoder outputs for Michel clustering"""
        try:
            hit_store = batch.get_node_store('hit')
            
            # Use instance decoder output edges
            if ('hit', 'cluster', 'particle') in batch.edge_types:
                return self._instance_based_regularization(batch, ('hit', 'cluster', 'particle'))
            
            # Fallback if no clustering available
            device = hit_store['x_raw'].device
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        except Exception as e:
            if self.verbose:
                print(f"   Regularization error: {e}")
            device = torch.device('cuda')
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def _instance_based_regularization(self, batch, edge_type):
        try:
            hit_store = batch.get_node_store('hit')
        
            # hit-to-instance mapping from instance decoder
            edge_index = batch[edge_type].edge_index
            hit_indices = edge_index[0]
            instance_indices = edge_index[1]

            #Check bounds
            max_hit_index = hit_store['x_raw'].size(0) - 1
            valid_mask = hit_indices <= max_hit_index

            if not torch.all(valid_mask):
                print(f"   WARNING: Found invalid hit indices, filtering them out")
                hit_indices = hit_indices[valid_mask]
                instance_indices = instance_indices[valid_mask]
        
            # Michel probabilities
            probabilities = torch.softmax(hit_store['x_semantic'], dim=1)
            michel_probs = probabilities[:, 3]
        
            # Process each instance
            unique_instances = torch.unique(instance_indices)
            total_reg_loss = torch.tensor(0.0, device=hit_store['x_raw'].device, requires_grad=True)
            michel_instances_found = 0
        
            for instance_id in unique_instances:
                instance_mask = instance_indices == instance_id
                instance_hits = hit_indices[instance_mask]

                if torch.any(instance_hits >= hit_store['x_raw'].size(0)):
                    print(f"   Skipping instance {instance_id} - invalid hit indices")
                    continue
            
                # Check if this instance is likely a Michel electron
                instance_michel_probs = michel_probs[instance_hits]
                avg_michel_prob = torch.mean(instance_michel_probs)

                #if michel_instances_found < 3:
                    #print(f"   Instance {instance_id}: {len(instance_hits)} hits, avg Michel prob: {avg_michel_prob.item():.3f}")
            
                if avg_michel_prob > 0.25:  # Michel electron threshold
                    michel_instances_found += 1
                
                    # Sum energy for complete Michel electron instance
                    instance_energies = hit_store['x_raw'][instance_hits, 2]
                    total_energy_raw = torch.sum(instance_energies)
                    total_energy_mev = total_energy_raw * 0.00580717  # Landau conversion
                
                    # Apply 30 MeV Gaussian penalty
                    target_energy = 30.0  # MeV per complete Michel electron
                
                    if total_energy_mev > 5.0:  # Only regularize reasonable energies
                        sigma = 10.0  # Tolerance around 30 MeV
                        prob_weight = avg_michel_prob  # Weight by Michel probability
                    
                        # Gaussian penalty around target
                        energy_diff = total_energy_mev - target_energy
                        gaussian_penalty = torch.exp(-0.5 * (energy_diff / sigma)**2)
                        instance_loss = self.lambda_param * prob_weight * (1 - gaussian_penalty) * 10
                    
                        total_reg_loss = total_reg_loss + instance_loss
                    
                        if self.verbose and michel_instances_found <= 3:  # Log first few
                            print(f"     Michel instance {instance_id}: {len(instance_hits)} hits")
                            print(f"     Avg Michel prob: {avg_michel_prob.item():.3f}")
                            print(f"     Total energy: {total_energy_mev.item():.2f} MeV")
                            print(f"     Loss contribution: {instance_loss.item():.6f}")
        
            #if michel_instances_found == 0:
                #print(f"   No instances with avg Michel prob > 0.3 found (checked {len(unique_instances)} instances)")
            #else:
                #print(f"   Found {michel_instances_found} Michel instances")
        
            if self.verbose:
                print(f"   Total regularization loss: {total_reg_loss.item():.6f}")
        
            return total_reg_loss
        
        except Exception as e:
            if self.verbose:
                print(f"   Instance regularization error: {e}")
            device = hit_store['x_raw'].device
            return torch.tensor(0.0, device=device, requires_grad=True)

def verify_gradient_flow(model, batch, michel_reg, verbose=True):
    """Verify that gradients flow through the regularization"""
    if verbose:
        print("GRADIENT FLOW VERIFICATION:")
        print("-" * 50)
    
    try:
        with torch.no_grad():
            hit_store = batch.get_node_store('hit')
            
            if verbose:
                print(f"  Semantic logits require_grad: {hit_store['x_semantic'].requires_grad}")
                print(f"  Regularizer callable: {callable(michel_reg)}")
        
        with torch.enable_grad():
            reg_loss = michel_reg(batch)
            
            if verbose:
                print(f"  Reg loss value: {reg_loss.item():.6f}")
                print(f"  Reg loss require_grad: {reg_loss.requires_grad}")
                print(f"  Reg loss is leaf: {reg_loss.is_leaf}")
            
            if reg_loss.requires_grad and reg_loss.item() > 0:
                has_grad_fn = reg_loss.grad_fn is not None
                
                if verbose:
                    print(f"  Tensor has grad_fn: {has_grad_fn}")
                    if has_grad_fn:
                        print(f"SUCCESS: Regularization is part of computational graph!")
                        print(f"Gradients will flow during training!")
                    else:
                        print(f"   Tensor is detached from computational graph")
                
                return has_grad_fn
            else:
                if verbose:
                    print(f" No gradients to test (zero loss or no grad)")
                return False
                
    except Exception as e:
        if verbose:
            print(f"Verification failed: {e}")
        return False
    finally:
        if verbose:
            print("-" * 50)

class LocalMetricsLogger:
    """Simplified metrics logging"""
    
    def __init__(self, output_dir, experiment_name):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.metrics_dir = self.output_dir / "metrics"
        self.logs_dir = self.output_dir / "logs"
        self.plots_dir = self.output_dir / "plots"
        
        # Create directories
        for dir_path in [self.metrics_dir, self.logs_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize files
        self.batch_metrics_file = self.metrics_dir / "batch_metrics.csv"
        self.epoch_metrics_file = self.metrics_dir / "epoch_metrics.csv"
        
        self.init_csv_files()
        self.setup_logging()
        
        # Metrics storage
        self.batch_metrics = []
        self.epoch_metrics = []
        
        # Tracking counters
        self.counters = {
            'total_batches': 0,
            'total_michel_pred': 0,
            'total_michel_gt': 0,
            'gradient_verified': False
        }
        
    def setup_logging(self):
        """Setup logging"""
        log_file = self.logs_dir / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__)
        
    def init_csv_files(self):
        """Initialize CSV files"""
        # Batch metrics
        with open(self.batch_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'epoch', 'batch', 'train_loss', 'michel_reg_loss', 
                           'total_loss', 'michel_pred', 'michel_gt', 'total_hits', 
                           'max_michel_prob', 'avg_michel_prob'])
        
        # Epoch metrics
        with open(self.epoch_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'epoch', 'avg_train_loss', 'avg_michel_reg_loss', 
                           'total_michel_pred', 'total_michel_gt', 'training_time_min'])
    
    def log_batch_metrics(self, epoch, batch_idx, train_loss, michel_reg_loss, total_loss, batch=None):
        """Log batch metrics"""
        timestamp = datetime.now().isoformat()
        self.counters['total_batches'] += 1
        
        # Initialize defaults
        michel_pred = 0
        michel_gt = 0
        total_hits = 0
        max_michel_prob = 0.0
        avg_michel_prob = 0.0
        
        if batch is not None:
            try:
                hit_store = batch.get_node_store('hit')
                
                with torch.no_grad():
                    # Get soft probabilities for monitoring
                    if 'x_semantic' in hit_store:
                        probs = torch.softmax(hit_store['x_semantic'], dim=1)
                        michel_probs = probs[:, 3]
                        max_michel_prob = torch.max(michel_probs).item()
                        avg_michel_prob = torch.mean(michel_probs).item()
                        
                        # Hard predictions for counting
                        y_pred = torch.argmax(hit_store['x_semantic'], dim=1)
                        michel_pred = torch.sum(y_pred == 3).item()
                        total_hits = len(y_pred)
                    
                    # Ground truth
                    if 'y_semantic' in hit_store:
                        y_true = hit_store['y_semantic']
                        michel_gt = torch.sum(y_true == 3).item()
                        
                        self.counters['total_michel_pred'] += michel_pred
                        self.counters['total_michel_gt'] += michel_gt
                        
            except Exception as e:
                self.logger.warning(f"Batch analysis failed: {e}")
        
        # Save to CSV
        with open(self.batch_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, epoch, batch_idx, train_loss, michel_reg_loss,
                           total_loss, michel_pred, michel_gt, total_hits,
                           max_michel_prob, avg_michel_prob])
        
        # Store in memory
        self.batch_metrics.append({
            'epoch': epoch, 'batch': batch_idx, 'train_loss': train_loss,
            'michel_reg_loss': michel_reg_loss, 'total_loss': total_loss,
            'michel_pred': michel_pred, 'michel_gt': michel_gt,
            'max_michel_prob': max_michel_prob, 'avg_michel_prob': avg_michel_prob
        })
        
        # Keep last 1000 batches to avoid memory issues
        if len(self.batch_metrics) > 1000:
            self.batch_metrics = self.batch_metrics[-500:]
    
    def log_epoch_metrics(self, epoch, avg_train_loss, avg_michel_reg_loss, training_time):
        """Log epoch metrics"""
        timestamp = datetime.now().isoformat()
        
        with open(self.epoch_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, epoch, avg_train_loss, avg_michel_reg_loss,
                           self.counters['total_michel_pred'], self.counters['total_michel_gt'],
                           training_time])
        
        self.epoch_metrics.append({
            'epoch': epoch, 'avg_train_loss': avg_train_loss,
            'avg_michel_reg_loss': avg_michel_reg_loss, 'training_time': training_time
        })
        
        # Generate plots
        self.generate_plots()
    
    def generate_plots(self):
        """training progress plots"""
        if not self.batch_metrics:
            return
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Extract data
            batches = [m['batch'] for m in self.batch_metrics]
            train_losses = [m['train_loss'] for m in self.batch_metrics]
            michel_losses = [m['michel_reg_loss'] for m in self.batch_metrics]
            michel_preds = [m['michel_pred'] for m in self.batch_metrics]
            michel_gts = [m['michel_gt'] for m in self.batch_metrics]
            max_probs = [m['max_michel_prob'] for m in self.batch_metrics]
            avg_probs = [m['avg_michel_prob'] for m in self.batch_metrics]
            
            # Training loss
            axes[0,0].plot(batches, train_losses, 'b-', alpha=0.7)
            axes[0,0].set_title('Training Loss')
            axes[0,0].set_xlabel('Batch')
            axes[0,0].set_ylabel('Loss')
            axes[0,0].grid(True, alpha=0.3)
            
            # Michel regularization loss
            axes[0,1].plot(batches, michel_losses, 'r-', alpha=0.7)
            axes[0,1].set_title('Michel Regularization Loss')
            axes[0,1].set_xlabel('Batch')
            axes[0,1].set_ylabel('Reg Loss')
            axes[0,1].grid(True, alpha=0.3)
            
            # Michel detection counts
            axes[0,2].plot(batches, michel_preds, 'g-', alpha=0.7, label='Predicted')
            axes[0,2].plot(batches, michel_gts, 'b--', alpha=0.7, label='Ground Truth')
            axes[0,2].set_title('Michel Detection Count')
            axes[0,2].set_xlabel('Batch')
            axes[0,2].set_ylabel('Count')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)
            
            # Michel probabilities
            axes[1,0].plot(batches, max_probs, 'purple', alpha=0.7, label='Max Prob')
            axes[1,0].plot(batches, avg_probs, 'orange', alpha=0.7, label='Avg Prob')
            axes[1,0].set_title('Michel Probabilities (Soft Classification)')
            axes[1,0].set_xlabel('Batch')
            axes[1,0].set_ylabel('Probability')
            axes[1,0].set_ylim(0, 1.1)
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # Combined losses
            total_losses = [m['total_loss'] for m in self.batch_metrics]
            axes[1,1].plot(batches, train_losses, 'b-', alpha=0.7, label='Classification')
            axes[1,1].plot(batches, michel_losses, 'r-', alpha=0.7, label='Michel Reg')
            axes[1,1].plot(batches, total_losses, 'k-', linewidth=2, label='Total')
            axes[1,1].set_title('Loss Breakdown')
            axes[1,1].set_xlabel('Batch')
            axes[1,1].set_ylabel('Loss')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
            # Epoch summary
            if self.epoch_metrics:
                epochs = [m['epoch'] for m in self.epoch_metrics]
                epoch_losses = [m['avg_train_loss'] for m in self.epoch_metrics]
                axes[1,2].plot(epochs, epoch_losses, 'bo-')
                axes[1,2].set_title('Average Loss per Epoch')
                axes[1,2].set_xlabel('Epoch')
                axes[1,2].set_ylabel('Avg Loss')
                axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            latest_epoch = max([m['epoch'] for m in self.batch_metrics]) if self.batch_metrics else 0
            plot_file = self.plots_dir / f"training_progress_epoch_{latest_epoch}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Plot saved: {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Plot generation failed: {e}")
            plt.close('all')
        
        summary_file = self.output_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📋 Summary saved: {summary_file}")
        return summary

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/nugraph/NG2-paper.gnn.h5')
    parser.add_argument('--output-dir', type=str, default='./training_outputs')
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--enable-michel-reg', action='store_true')
    parser.add_argument('--michel-reg-lambda', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--verify-gradients', action='store_true')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    return parser.parse_args()

def train(args):
    """Main training function with DIRECT clustering"""
    try:
        # Create experiment directory
        experiment_dir = Path(args.output_dir) / args.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        metrics_logger = LocalMetricsLogger(experiment_dir, args.experiment_name)
        
        metrics_logger.logger.info(f"DIRECT CLUSTERING training: {args.experiment_name}")
        
        # Import libraries
        import nugraph
        from nugraph.data import H5DataModule
        from nugraph.models.nugraph3.decoders.spacepoint import SpacepointDecoder
        import nugraph.models.nugraph3.decoders
        nugraph.models.nugraph3.decoders.SpacepointDecoder = SpacepointDecoder
        from nugraph.models.nugraph3.nugraph3 import NuGraph3
        
        # Load data
        metrics_logger.logger.info(f'Loading data: {args.data_path}')
        nudata = H5DataModule(args.data_path, batch_size=args.batch_size)
        transform = NuGraph3.transform(nudata.planes)
        nudata.transform = transform
        print(f"Transform applied for planes: {nudata.planes}")

        #print("Debugging data structure...")
        test_batch = next(iter(nudata.val_dataloader()))
        #print(f"Available node types: {list(test_batch.node_types)}")
        #print(f"Available edge types: {list(test_batch.edge_types)}")

        # Check what's in the batch
        for node_type in test_batch.node_types:
            node_store = test_batch.get_node_store(node_type)
            print(f"Node '{node_type}': {node_store.num_nodes} nodes")
            if hasattr(node_store, 'x') and node_store.x is not None:
                print(f"  Features shape: {node_store.x.shape}")

        def fix_batch_structure(batch):
            """Fix batch structure for all node types including sp"""
            try:
                # Ensure all node types have proper batch metadata
                for node_type in batch.node_types:
                    node_store = batch.get_node_store(node_type)
                
                    # Initialize features for empty node types
                    if node_type == 'sp' and (not hasattr(node_store, 'x') or node_store.x is None or node_store.x.size(1) == 0):
                        device = batch.get_node_store('hit').x.device
                        node_store.x = torch.zeros(node_store.num_nodes, 16, device=device)
                
                    elif node_type == 'evt' and (not hasattr(node_store, 'x') or node_store.x is None or node_store.x.size(1) == 0):
                        device = batch.get_node_store('hit').x.device  
                        node_store.x = torch.zeros(node_store.num_nodes, 16, device=device)
                    
                    elif node_type == 'particle-truth' and (not hasattr(node_store, 'x') or node_store.x is None or node_store.x.size(1) == 0):
                        device = batch.get_node_store('hit').x.device
                        node_store.x = torch.zeros(node_store.num_nodes, 8, device=device)
                
                    # Ensure proper batch slicing
                    if hasattr(batch, '_slice_dict') and node_type not in batch._slice_dict:
                        if hasattr(node_store, 'ptr'):
                            batch._slice_dict[node_type] = {'x': node_store.ptr}
                        else:
                            device = batch.get_node_store('hit').x.device
                            node_store.ptr = torch.arange(0, node_store.num_nodes + 1, 
                                                         node_store.num_nodes // batch.num_graphs, 
                                                         device=device, dtype=torch.long)
                            batch._slice_dict[node_type] = {'x': node_store.ptr}
                
                    # Ensure proper increment dict
                    if hasattr(batch, '_inc_dict') and node_type not in batch._inc_dict:
                        device = batch.get_node_store('hit').x.device
                        batch._inc_dict[node_type] = {'x': torch.zeros(batch.num_graphs, device=device, dtype=torch.long)}
                            
            except Exception as e:
                print(f"Batch fixing error: {e}")
        
            return batch

        # the fix is applied to all dataloaders
        original_train_dataloader = nudata.train_dataloader
        original_val_dataloader = nudata.val_dataloader
        original_test_dataloader = nudata.test_dataloader

        def fixed_train_dataloader():
            loader = original_train_dataloader()
            class FixedLoader:
                def __init__(self, original_loader):
                    self.original_loader = original_loader
                def __iter__(self):
                    for batch in self.original_loader:
                        yield fix_batch_structure(batch)
                def __len__(self):
                    return len(self.original_loader)
            return FixedLoader(loader)

        def fixed_val_dataloader():
            loader = original_val_dataloader()
            class FixedLoader:
                def __init__(self, original_loader):
                    self.original_loader = original_loader
                def __iter__(self):
                    for batch in self.original_loader:
                        yield fix_batch_structure(batch)
                def __len__(self):
                    return len(self.original_loader)
            return FixedLoader(loader)

        nudata.train_dataloader = fixed_train_dataloader
        nudata.val_dataloader = fixed_val_dataloader

        # Create model
        metrics_logger.logger.info('Creating NuGraph3 model')
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
            event_head=False,
            semantic_head=True,
            filter_head=True,
            vertex_head=False,
            instance_head=True,  # ENABLED 
            spacepoint_head=False,
            use_checkpointing=True,
            lr=args.learning_rate
        )
        
        def fix_sp_node_features(batch):
            """Initialize empty sp node features to prevent KeyError"""
            try:
                sp_store = batch.get_node_store('sp')
                if not hasattr(sp_store, 'x') or sp_store.x is None or sp_store.x.size(1) == 0:
                    # Initialize with nexus_features size (16)
                    device = batch.get_node_store('hit').x.device
                    sp_store.x = torch.zeros(sp_store.num_nodes, 16, device=device)
            except Exception as e:
                print(f"Warning: Could not fix sp node features: {e}")
            return batch
        
        original_forward = model.forward
        
        def fixed_forward(data, stage=None):
            """Forward pass with sp node fix"""
            data = fix_sp_node_features(data)
            return original_forward(data, stage)
        
        model.forward = fixed_forward

        original_validation_step = model.validation_step

        def fixed_validation_step(batch, batch_idx):
            """Validation step with comprehensive sp node fix"""
            try:
                # Fix sp node features AND batch slicing
                sp_store = batch.get_node_store('sp')
                
                # Initialize features if missing
                if not hasattr(sp_store, 'x') or sp_store.x is None or sp_store.x.size(1) == 0:
                    device = batch.get_node_store('hit').x.device
                    sp_store.x = torch.zeros(sp_store.num_nodes, 16, device=device)
                
                # Fix batch slicing for sp nodes
                if hasattr(batch, '_slice_dict') and 'sp' not in batch._slice_dict:
                    # Create proper slice dict entry for sp nodes
                    batch._slice_dict['sp'] = {'x': sp_store.ptr if hasattr(sp_store, 'ptr') else torch.tensor([0, sp_store.num_nodes], device=device)}
                    
                # Fix increment dict for sp nodes  
                if hasattr(batch, '_inc_dict') and 'sp' not in batch._inc_dict:
                    batch._inc_dict['sp'] = {'x': torch.zeros(batch.num_graphs, device=device, dtype=torch.long)}
                
                return original_validation_step(batch, batch_idx)
                
            except Exception as e:
                print(f"Validation step error: {e}")
                # Skip validation if it fails
                return torch.tensor(0.0, device=batch.get_node_store('hit').x.device)
        
        model.validation_step = fixed_validation_step
        
        if torch.cuda.is_available():
            model = model.cuda()
            metrics_logger.logger.info("Model on GPU")
        
        # Setup Michel regularization
        if args.enable_michel_reg:
            michel_reg = MichelRegularizer(lambda_param=args.michel_reg_lambda, verbose=False)
            metrics_logger.logger.info(f'DIRECT CLUSTERING Michel reg enabled (λ={args.michel_reg_lambda})')
        else:
            michel_reg = None
            metrics_logger.logger.info('Baseline training (no regularization)')
        
        # Training state
        original_training_step = model.training_step
        batch_losses = []
        michel_losses = []
        epoch_start_time = time.time()
        gradient_verified = False
        
        def enhanced_training_step(batch, batch_idx):
            """Enhanced training step with DIRECT clustering"""
            nonlocal gradient_verified
            
            try:
                if torch.cuda.is_available():
                    batch = batch.to('cuda')
                
                # Original training step
                loss = original_training_step(batch, batch_idx)

                for node_type in batch.node_types:
                    if 'particle' in node_type.lower() or 'instance' in node_type.lower():
                        node_store = batch.get_node_store(node_type)
                        #print(f"  Node type '{node_type}': {node_store.num_nodes} nodes")

                for edge_type in batch.edge_types:
                    if 'particle' in str(edge_type) and 'truth' not in str(edge_type):
                        #print(f"  Found non-truth particle edge: {edge_type}")
                        edge_store = batch.get_edge_store(*edge_type)
                        #print(f"    Edge count: {edge_store.edge_index.size(1)}")
                
                # Apply DIRECT clustering Michel regularization
                reg_loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
                if args.enable_michel_reg and michel_reg:
                    try:
                        reg_loss = michel_reg(batch)
                        total_loss = loss + reg_loss
                        
                        # Gradient verification on first active batch
                        if args.verify_gradients and not gradient_verified and reg_loss.item() > 0:
                            metrics_logger.logger.info(f"\n GRADIENT VERIFICATION (Batch {batch_idx}):")
                            success = verify_gradient_flow(model, batch, michel_reg, verbose=True)
                            metrics_logger.counters['gradient_verified'] = success
                            if success:
                                metrics_logger.logger.info(" GRADIENT FLOW: SUCCESS!")
                            else:
                                metrics_logger.logger.warning("GRADIENT FLOW: FAILED!")
                            gradient_verified = True
                    
                        model.log('michel_reg_loss', reg_loss, batch_size=getattr(batch, 'num_graphs', 1))
                        
                    except Exception as e:
                        metrics_logger.logger.warning(f"Regularization error: {e}")
                        total_loss = loss
                else:
                    total_loss = loss
                
                # Extract loss values
                loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
                reg_loss_val = reg_loss.item() if hasattr(reg_loss, 'item') else float(reg_loss)
                total_loss_val = total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
                
                # Track metrics
                batch_losses.append(loss_val)
                michel_losses.append(reg_loss_val)
                
                # Get current epoch
                current_epoch = getattr(model.trainer, 'current_epoch', 0) if hasattr(model, 'trainer') else 0
                
                # Log metrics
                metrics_logger.log_batch_metrics(
                    current_epoch, batch_idx, loss_val, reg_loss_val, total_loss_val, batch
                )
                
                # Detailed logging
                if batch_idx % args.log_interval == 0:
                    latest = metrics_logger.batch_metrics[-1] if metrics_logger.batch_metrics else {}
                    michel_pred = latest.get('michel_pred', 0)
                    michel_gt = latest.get('michel_gt', 0)
                    max_prob = latest.get('max_michel_prob', 0)
                    avg_prob = latest.get('avg_michel_prob', 0)
                    
                    if args.enable_michel_reg:
                        metrics_logger.logger.info(
                            f"Epoch {current_epoch}, Batch {batch_idx}: "
                            f"Loss={loss_val:.4f}, Michel reg={reg_loss_val:.6f}, "
                            f"Michel pred/gt={michel_pred}/{michel_gt}, "
                            f"Max/Avg prob={max_prob:.3f}/{avg_prob:.3f}"
                        )
                    else:
                        metrics_logger.logger.info(
                            f"Epoch {current_epoch}, Batch {batch_idx}: "
                            f"Loss={loss_val:.4f}, Michel pred/gt={michel_pred}/{michel_gt}"
                        )
                
                return total_loss
                
            except Exception as e:
                metrics_logger.logger.error(f"Training step failed: {e}")
                return torch.tensor(1.0, device=next(model.parameters()).device, requires_grad=True)
        
        def on_epoch_end():
            """Epoch end callback"""
            nonlocal epoch_start_time
            
            current_epoch = getattr(model.trainer, 'current_epoch', 0) if hasattr(model, 'trainer') else 0
            epoch_time = time.time() - epoch_start_time
            
            avg_train_loss = np.mean(batch_losses) if batch_losses else 0
            avg_michel_loss = np.mean(michel_losses) if michel_losses else 0
            
            metrics_logger.log_epoch_metrics(current_epoch, avg_train_loss, avg_michel_loss, epoch_time / 60)
            
            metrics_logger.logger.info(
                f"📈 Epoch {current_epoch}: "
                f"Avg Loss={avg_train_loss:.4f}, "
                f"Avg Michel Reg={avg_michel_loss:.6f}, "
                f"Time={epoch_time/60:.1f}min"
            )
            
            # Reset for next epoch
            batch_losses.clear()
            michel_losses.clear()
            epoch_start_time = time.time()
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        # Replace methods
        model.training_step = enhanced_training_step
        original_epoch_end = getattr(model, 'on_train_epoch_end', lambda: None)
        model.on_train_epoch_end = lambda: (original_epoch_end(), on_epoch_end())
        
        # Setup checkpointing
        checkpoint_dir = experiment_dir / "checkpoints"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch}-{loss/train:.4f}',
            save_top_k=3,
            monitor='loss/train',
            mode='min',
            save_last=True
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[checkpoint_callback],
            logger=False,
            enable_progress_bar=True,
            log_every_n_steps=args.log_interval,
            deterministic=True
        )
        
        # Start training
        metrics_logger.logger.info(f" Starting CLUSTERING training for {args.epochs} epochs")
        start_time = time.time()
        
        if args.resume_from_checkpoint:
            trainer.fit(model, nudata, ckpt_path=args.resume_from_checkpoint)
        else:
            trainer.fit(model, nudata)
        
        total_time = time.time() - start_time
        metrics_logger.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        # Save summary
        model_info = {
            'model_type': 'NuGraph3',
            'epochs_trained': args.epochs,
            'michel_regularization': args.enable_michel_reg,
            'michel_reg_lambda': args.michel_reg_lambda if args.enable_michel_reg else None,
            'total_training_time_hours': total_time / 3600,
            'gradient_flow_verified': metrics_logger.counters['gradient_verified'],
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'clustering_method': 'Direct DBSCAN',
            'instance_decoder_enabled': False
        }
        
        final_summary = metrics_logger.save_final_summary(model_info)
        
        # Final summary
        metrics_logger.logger.info(" FINAL DIRECT CLUSTERING TRAINING SUMMARY:")
        metrics_logger.logger.info("=" * 60)
        physics_val = final_summary.get('physics_validation', {})
        metrics_logger.logger.info(f" Gradient flow verified: {physics_val.get('gradient_flow_verified', False)}")
        metrics_logger.logger.info(f"Total Michel predicted: {physics_val.get('total_michel_predicted', 0)}")
        metrics_logger.logger.info(f"Total Michel ground truth: {physics_val.get('total_michel_ground_truth', 0)}")
        metrics_logger.logger.info(f"Detection ratio: {physics_val.get('detection_ratio', 0):.3f}")
        metrics_logger.logger.info("=" * 60)
        
        metrics_logger.logger.info(f" All outputs saved to: {experiment_dir}")
        return model, trainer

        print(model)
        
    except Exception as e:
        print(f" Training failed: {e}")
        raise

if __name__ == '__main__':
    print("Michel Electron Training")
    print("Instance Decoder CLUSTERING APPROACH:")
    print("GROUP MICHEL HITS → SUM ENERGIES → APPLY 30 MeV PENALTY!")
    
    try:
        args = configure()
        
        print(f"Experiment: {args.experiment_name}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Gradient verification: {'ENABLED' if args.verify_gradients else 'Available'}")
        print("=" * 70)
        
        # Train the model
        model, trainer = train(args)

        if model is not None and trainer is not None:
            print("CLUSTERING training completed successfully!")
            print(f"Check results in: training_outputs/{args.experiment_name}/")
            print("Key outputs:")
            print("   - plots/training_progress_epoch_*.png")
            print("   - metrics/batch_metrics.csv")
            print("   - training_summary.json")
            print("\n GRADIENT FLOW VERIFICATION:")
        else:
            print("Training failed - check logs for details")
            
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
    except Exception as e:
        print(f" Training failed: {e}")
        import traceback
        traceback.print_exc()

