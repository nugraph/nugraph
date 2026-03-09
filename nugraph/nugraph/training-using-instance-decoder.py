#!/usr/bin/env python

import os
import sys
import argparse
import torch
import pytorch_lightning as pl
import json
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix
import gc
import tqdm


# Environment setup
os.environ['WANDB_MODE'] = 'disabled'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

def cleanup_memory():
    """Comprehensive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def cleanup_batch_memory(batch):
    """Clean up batch-specific memory"""
    try:
        for node_type in batch.node_types:
            node_store = batch.get_node_store(node_type)
            if hasattr(node_store, 'x') and node_store.x is not None:
                if node_store.x.grad is not None:
                    node_store.x.grad = None
        
        for edge_type in batch.edge_types:
            edge_store = batch.get_edge_store(*edge_type)
            if hasattr(edge_store, 'edge_index') and edge_store.edge_index.grad is not None:
                edge_store.edge_index.grad = None
                
    except Exception as e:
        print(f"Batch cleanup warning: {e}")

def setup_nugraph_environment():
    """Setup nugraph environment variables"""
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

setup_nugraph_environment()
plt.switch_backend('Agg')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MemoryCleanupCallback(pl.Callback):
    def __init__(self, metrics_logger):
        super().__init__()
        self.metrics_logger = metrics_logger
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Reset epoch metrics at start of each epoch"""
        self.metrics_logger.reset_epoch_metrics()
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log epoch summary and cleanup memory"""
        self.metrics_logger.log_epoch_summary(trainer.current_epoch)
        cleanup_memory()
        print(f"Epoch {trainer.current_epoch} completed - memory cleaned")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        cleanup_memory()


class CorrectedMichelEnergyAnalyzer:
    """Analyzer with overlap filtering and diagnostics - NO THRESHOLDS"""

    def __init__(self, overlap_threshold=0.5, verbose=False):
        self.overlap_threshold = overlap_threshold
        self.verbose = verbose
        self.true_clusters = []
        self.predicted_clusters = []
        self.validated_predicted_clusters = []

        self.all_true_labels = []
        self.all_pred_labels = []

    def analyze_batch(self, batch, predictions=None):
        """Analyze true, predicted, and validated predicted Michel clusters"""

        if predictions is not None:
            self._collect_labels_for_confusion_matrix(batch, predictions)

        if ('hit', 'cluster-truth', 'particle-truth') not in batch.edge_types:
            if self.verbose:
                print("   No ground truth clustering edges found")
            return

        true_clusters = self._analyze_true_clusters_with_diagnostics(batch, predictions)
        self.true_clusters.extend(true_clusters)

        if predictions is not None:
            pred_clusters = self._analyze_predicted_clusters_with_instance_decoder(batch, predictions)
            self.predicted_clusters.extend(pred_clusters)

            validated_clusters = self._filter_predicted_by_overlap(batch, pred_clusters, true_clusters)
            self.validated_predicted_clusters.extend(validated_clusters)

        if self.verbose:
            print(f"   True: {len(true_clusters)}, "
                  f"Predicted: {len(pred_clusters) if predictions is not None else 0}, "
                  f"Validated: {len(validated_clusters) if predictions is not None else 0}")

    def _collect_labels_for_confusion_matrix(self, batch, predictions):
        """Collect true and predicted labels for confusion matrix"""
        try:
            hit_store = batch.get_node_store('hit')

            if 'y_semantic' not in hit_store:
                return

            y_true = hit_store['y_semantic'].cpu().numpy()
            y_pred = torch.argmax(predictions, dim=1).cpu().numpy()

            self.all_true_labels.extend(y_true.tolist())
            self.all_pred_labels.extend(y_pred.tolist())

        except Exception as e:
            if self.verbose:
                print(f"Label collection error: {e}")

    def _analyze_true_clusters_with_diagnostics(self, batch, predictions=None):
        """Analyze true Michel clusters and track their predicted probabilities"""
        try:
            hit_store = batch.get_node_store('hit')

            truth_edge_type = ('hit', 'cluster-truth', 'particle-truth')
            if 'y_semantic' not in hit_store:
                return []

            y_semantic = hit_store['y_semantic']
            edge_index = batch[truth_edge_type].edge_index
            hit_indices = edge_index[0]
            particle_indices = edge_index[1]

            if predictions is not None:
                probabilities = torch.softmax(predictions, dim=1)
            else:
                probabilities = None

            michel_mask = y_semantic == 3
            michel_hit_indices = torch.where(michel_mask)[0]

            if len(michel_hit_indices) == 0:
                return []

            michel_particles = set()
            for hit_idx in michel_hit_indices:
                particle_mask = hit_indices == hit_idx
                if torch.any(particle_mask):
                    particle_id = particle_indices[particle_mask][0]
                    michel_particles.add(particle_id.item())

            clusters = []

            for particle_id in michel_particles:
                particle_mask = particle_indices == particle_id
                particle_hit_indices = hit_indices[particle_mask]

                particle_energies = hit_store['x_raw'][particle_hit_indices, 2]
                total_energy_mev = torch.sum(particle_energies) * 0.00580717

                particle_semantic_labels = y_semantic[particle_hit_indices]
                michel_fraction = torch.sum(particle_semantic_labels == 3).float() / len(particle_hit_indices)

                predicted_class_probs = {'MIP': 0.0, 'HIP': 0.0, 'Shower': 0.0, 'Michel': 0.0, 'Diffuse': 0.0}
                predicted_max_class = 'Unknown'
                would_be_predicted = False

                if probabilities is not None:
                    particle_probs = probabilities[particle_hit_indices]
                    avg_class_probs = torch.mean(particle_probs, dim=0)

                    predicted_class_probs = {
                        'MIP': avg_class_probs[0].item(),
                        'HIP': avg_class_probs[1].item(),
                        'Shower': avg_class_probs[2].item(),
                        'Michel': avg_class_probs[3].item(),
                        'Diffuse': avg_class_probs[4].item()
                    }

                    max_prob_class_idx = torch.argmax(avg_class_probs)
                    class_names = ['MIP', 'HIP', 'Shower', 'Michel', 'Diffuse']
                    predicted_max_class = class_names[max_prob_class_idx.item()]
                    would_be_predicted = (max_prob_class_idx == 3)

                cluster_info = {
                    'cluster_id': f'true_{particle_id}',
                    'energy_mev': total_energy_mev.item(),
                    'num_hits': len(particle_hit_indices),
                    'michel_fraction': michel_fraction.item(),
                    'predicted_class_probs': predicted_class_probs,
                    'predicted_max_class': predicted_max_class,
                    'would_be_predicted': would_be_predicted,
                    'cluster_type': 'true',
                    'hit_indices': particle_hit_indices.cpu().numpy()
                }

                clusters.append(cluster_info)

            return clusters

        except Exception as e:
            if self.verbose:
                print(f"True cluster analysis error: {e}")
            return []

    def _filter_predicted_by_overlap(self, batch, pred_clusters, true_clusters_in_batch):
        """Filter predicted clusters by overlap with true Michel particles"""
        try:
            if not pred_clusters or not true_clusters_in_batch:
                return []

            current_batch_true_michel_hits = set()
            for true_cluster in true_clusters_in_batch:
                if 'hit_indices' in true_cluster:
                    current_batch_true_michel_hits.update(true_cluster['hit_indices'])

            validated_clusters = []

            for pred_cluster in pred_clusters:
                if 'hit_indices' not in pred_cluster:
                    continue

                pred_hits = set(pred_cluster['hit_indices'])

                overlap_hits = pred_hits.intersection(current_batch_true_michel_hits)
                overlap_fraction = len(overlap_hits) / len(pred_hits) if len(pred_hits) > 0 else 0.0

                if overlap_fraction >= self.overlap_threshold:
                    validated_cluster = pred_cluster.copy()
                    validated_cluster['overlap_fraction'] = overlap_fraction
                    validated_cluster['cluster_type'] = 'validated_predicted'
                    validated_clusters.append(validated_cluster)

            return validated_clusters

        except Exception as e:
            if self.verbose:
                print(f"Overlap filtering error: {e}")
            return []

    def _analyze_predicted_clusters_with_instance_decoder(self, batch, predictions):
        """Analyze predicted Michel clusters using max probability approach"""
        try:
            hit_store = batch.get_node_store('hit')

            edge_type = ('hit', 'cluster', 'particle')
            if edge_type not in batch.edge_types:
                return []

            edge_index = batch[edge_type].edge_index
            hit_indices = edge_index[0]
            instance_indices = edge_index[1]

            max_hit_index = hit_store['x_raw'].size(0) - 1
            if hit_indices.numel() == 0 or instance_indices.numel() == 0:
                return []

            valid_hit_mask = (hit_indices >= 0) & (hit_indices <= max_hit_index)
            if not torch.all(valid_hit_mask):
                hit_indices = hit_indices[valid_hit_mask]
                instance_indices = instance_indices[valid_hit_mask]

            if hit_indices.numel() == 0:
                return []

            probabilities = torch.softmax(predictions, dim=1)

            unique_instances = torch.unique(instance_indices)
            clusters = []

            for instance_id in unique_instances:
                instance_mask = instance_indices == instance_id
                instance_hits = hit_indices[instance_mask]

                if torch.any(instance_hits >= hit_store['x_raw'].size(0)):
                    continue

                instance_probs = probabilities[instance_hits]
                avg_class_probs = torch.mean(instance_probs, dim=0)

                max_prob_class = torch.argmax(avg_class_probs)
                michel_class_idx = 3

                if max_prob_class == michel_class_idx:
                    avg_michel_prob = avg_class_probs[michel_class_idx]

                    instance_energies = hit_store['x_raw'][instance_hits, 2]
                    total_energy_raw = torch.sum(instance_energies)
                    total_energy_mev = total_energy_raw * 0.00580717

                    true_michel_fraction = 0.0
                    if 'y_semantic' in hit_store:
                        instance_gt_labels = hit_store['y_semantic'][instance_hits]
                        true_michel_fraction = torch.sum(instance_gt_labels == 3).float() / len(instance_hits)

                    cluster_info = {
                        'cluster_id': f'pred_{instance_id.item()}',
                        'energy_mev': total_energy_mev.item(),
                        'num_hits': len(instance_hits),
                        'avg_michel_prob': avg_michel_prob.item(),
                        'all_class_probs': {
                            'MIP': avg_class_probs[0].item(),
                            'HIP': avg_class_probs[1].item(),
                            'Shower': avg_class_probs[2].item(),
                            'Michel': avg_class_probs[3].item(),
                            'Diffuse': avg_class_probs[4].item()
                        },
                        'true_michel_fraction': true_michel_fraction.item(),
                        'cluster_type': 'predicted',
                        'hit_indices': instance_hits.cpu().numpy()
                    }

                    clusters.append(cluster_info)

            return clusters

        except Exception as e:
            if self.verbose:
                print(f"Predicted cluster analysis error: {e}")
            return []

    def get_energy_data(self):
        """Get energy data for three-way plotting"""
        true_energies = [c['energy_mev'] for c in self.true_clusters]
        pred_energies = [c['energy_mev'] for c in self.predicted_clusters]
        validated_energies = [c['energy_mev'] for c in self.validated_predicted_clusters]
        
        return true_energies, pred_energies, validated_energies
    
    def plot_energy_spectra(self, output_dir):
        """Create three-panel energy spectrum plot"""
        true_energies, pred_energies, validated_energies = self.get_energy_data()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        if true_energies:
            axes[0].hist(true_energies, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0].axvline(40, color='red', linestyle='--', linewidth=2, label='Target: 40 MeV')
            axes[0].set_xlabel('Energy (MeV)')
            axes[0].set_ylabel('Count')
            axes[0].set_title(f'True Michel Energy Spectrum\n(n={len(true_energies)})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            mean_energy = np.mean(true_energies)
            std_energy = np.std(true_energies)
            axes[0].text(0.05, 0.95, f'Mean: {mean_energy:.1f} MeV\nStd: {std_energy:.1f} MeV', 
                        transform=axes[0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[0].text(0.5, 0.5, 'No true Michel clusters', 
                        transform=axes[0].transAxes, ha='center', va='center')
            axes[0].set_title('True Michel Energy Spectrum\n(n=0)')
        
        if pred_energies:
            axes[1].hist(pred_energies, bins=50, alpha=0.7, color='orange', edgecolor='black')
            axes[1].axvline(40, color='red', linestyle='--', linewidth=2, label='Target: 40 MeV')
            axes[1].set_xlabel('Energy (MeV)')
            axes[1].set_ylabel('Count')
            axes[1].set_title(f'Predicted Michel Energy Spectrum\n(n={len(pred_energies)})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            mean_energy = np.mean(pred_energies)
            std_energy = np.std(pred_energies)
            axes[1].text(0.05, 0.95, f'Mean: {mean_energy:.1f} MeV\nStd: {std_energy:.1f} MeV', 
                        transform=axes[1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[1].text(0.5, 0.5, 'No predicted Michel clusters', 
                        transform=axes[1].transAxes, ha='center', va='center')
            axes[1].set_title('Predicted Michel Energy Spectrum\n(n=0)')
        
        if validated_energies:
            axes[2].hist(validated_energies, bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[2].axvline(40, color='red', linestyle='--', linewidth=2, label='Target: 40 MeV')
            axes[2].set_xlabel('Energy (MeV)')
            axes[2].set_ylabel('Count')
            axes[2].set_title(f'Validated Michel Energy Spectrum\n(≥{self.overlap_threshold*100:.0f}% overlap, n={len(validated_energies)})')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            mean_energy = np.mean(validated_energies)
            std_energy = np.std(validated_energies)
            axes[2].text(0.05, 0.95, f'Mean: {mean_energy:.1f} MeV\nStd: {std_energy:.1f} MeV', 
                        transform=axes[2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[2].text(0.5, 0.5, 'No validated Michel clusters', 
                        transform=axes[2].transAxes, ha='center', va='center')
            axes[2].set_title(f'Validated Michel Energy Spectrum\n(≥{self.overlap_threshold*100:.0f}% overlap, n=0)')
        
        plt.tight_layout()
        
        energy_plot_path = Path(output_dir) / "michel_energy_spectra.png"
        plt.savefig(energy_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Energy spectra plot saved: {energy_plot_path}")
        return energy_plot_path
    
    def plot_confusion_matrix(self, output_dir):
        """Create confusion matrix plot"""
        if not self.all_true_labels or not self.all_pred_labels:
            print("No labels available for confusion matrix")
            return None
        
        class_names = ['MIP', 'HIP', 'Shower', 'Michel', 'Diffuse']
        
        cm = confusion_matrix(self.all_true_labels, self.all_pred_labels, 
                            labels=list(range(len(class_names))))
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', 
                   xticklabels=class_names, yticklabels=class_names,
                   cmap='Blues', cbar_kws={'label': 'Fraction'})
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Semantic Classification Confusion Matrix\n(Normalized by True Class)')
        
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j+0.5, i+0.7, f'({cm[i,j]})', 
                        ha='center', va='center', fontsize=8, color='gray')
        
        plt.tight_layout()
        
        cm_plot_path = Path(output_dir) / "confusion_matrix.png"
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix plot saved: {cm_plot_path}")
        return cm_plot_path
    
    def print_diagnostics(self):
        """Print comprehensive diagnostic information"""
        print(f"\nMichel Cluster Diagnostics:")
        print(f"   Total true Michel particles: {len(self.true_clusters)}")
        print(f"   Total predicted clusters: {len(self.predicted_clusters)}")
        print(f"   Validated predicted clusters: {len(self.validated_predicted_clusters)}")
        
        if self.true_clusters:
            would_predict = [c['would_be_predicted'] for c in self.true_clusters]
            predicted_classes = [c['predicted_max_class'] for c in self.true_clusters]
            
            print(f"\nTrue Michel Prediction Analysis:")
            print(f"   Would be predicted as Michel: {sum(would_predict)}/{len(would_predict)} ({100*sum(would_predict)/len(would_predict):.1f}%)")
            
            from collections import Counter
            class_counts = Counter(predicted_classes)
            print(f"   Predicted classes for true Michel particles:")
            for class_name, count in class_counts.items():
                percentage = 100 * count / len(predicted_classes)
                print(f"     {class_name}: {count}/{len(predicted_classes)} ({percentage:.1f}%)")
            
            michel_probs = [c['predicted_class_probs']['Michel'] for c in self.true_clusters]
            print(f"   Average Michel probability for true Michel particles: {np.mean(michel_probs):.3f}")
            
        if self.validated_predicted_clusters:
            overlaps = [c['overlap_fraction'] for c in self.validated_predicted_clusters]
            print(f"\nValidated Cluster Overlap Analysis:")
            print(f"   Average overlap: {np.mean(overlaps):.2f}")
            print(f"   Overlap range: {np.min(overlaps):.2f} - {np.max(overlaps):.2f}")


class ImprovedMichelRegularizer:
    """
    - Target energy: 40 MeV
    - Max probability approach (NO THRESHOLDS)
    - Anti-fragmentation penalty 
    - Strong penalty for high-energy clusters (≥100 MeV)
    """
    def __init__(self, lambda_param=0.1, target_energy=40.0, high_energy_threshold=100.0, verbose=False):
        self.lambda_param = lambda_param
        self.target_energy = target_energy
        self.high_energy_threshold = high_energy_threshold
        self.verbose = verbose

    def __call__(self, batch):
        """Apply improved Michel physics regularization"""
        try:
            hit_store = batch.get_node_store('hit')
            
            if ('hit', 'cluster', 'particle') in batch.edge_types:
                return self._improved_regularization(batch, ('hit', 'cluster', 'particle'))
            
            device = hit_store['x_raw'].device
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        except Exception as e:
            if self.verbose:
                print(f"   Regularization error: {e}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def _improved_regularization(self, batch, edge_type):
        """Apply improved physics constraints with max probability approach and anti-fragmentation"""
        try:
            hit_store = batch.get_node_store('hit')
        
            edge_index = batch[edge_type].edge_index
            hit_indices = edge_index[0]
            instance_indices = edge_index[1]

            max_hit_index = hit_store['x_raw'].size(0) - 1
            valid_mask = hit_indices <= max_hit_index
            if not torch.all(valid_mask):
                hit_indices = hit_indices[valid_mask]
                instance_indices = instance_indices[valid_mask]
        
            probabilities = torch.softmax(hit_store['x_semantic'], dim=1)
            
            y_true = hit_store.get('y_semantic', None)
        
            unique_instances = torch.unique(instance_indices)
            total_reg_loss = torch.tensor(0.0, device=hit_store['x_raw'].device, requires_grad=True)
            michel_instances_processed = 0
            
            # Anti-fragmentation loss
            fragmentation_loss = torch.tensor(0.0, device=hit_store['x_raw'].device, requires_grad=True)
            
            if y_true is not None and ('hit', 'cluster-truth', 'particle-truth') in batch.edge_types:
                truth_edge_type = ('hit', 'cluster-truth', 'particle-truth')
                truth_edge_index = batch[truth_edge_type].edge_index
                truth_hit_indices = truth_edge_index[0]
                truth_particle_indices = truth_edge_index[1]
                
                michel_mask = y_true == 3
                michel_hit_indices_set = set(torch.where(michel_mask)[0].cpu().numpy().tolist())
                
                michel_particles = set()
                for hit_idx in michel_hit_indices_set:
                    if hit_idx < 0 or hit_idx > max_hit_index:
                        continue
                    hit_idx_tensor = torch.tensor([hit_idx], device=truth_hit_indices.device, dtype=truth_hit_indices.dtype)
                    particle_mask = truth_hit_indices == hit_idx_tensor
                    if torch.any(particle_mask):
                        particle_id = truth_particle_indices[particle_mask][0]
                        michel_particles.add(particle_id.item())
                
                for true_particle_id in michel_particles:
                    particle_mask = truth_particle_indices == true_particle_id
                    true_particle_hit_indices = truth_hit_indices[particle_mask]
                    
                    if len(true_particle_hit_indices) == 0:
                        continue
                    
                    instance_assignments = []
                    for true_hit in true_particle_hit_indices:
                        true_hit_item = true_hit.item()
                        if true_hit_item < 0 or true_hit_item > max_hit_index:
                            continue
                        pred_mask = hit_indices == true_hit
                        if torch.any(pred_mask):
                            assigned_instance = instance_indices[pred_mask][0]
                            instance_assignments.append(assigned_instance.item())
                    
                    if len(instance_assignments) > 0:
                        num_unique_instances = len(set(instance_assignments))
                        
                        if num_unique_instances > 1:
                            frag_penalty = self.lambda_param * (num_unique_instances - 1) * 2.0
                            fragmentation_loss = fragmentation_loss + frag_penalty
                            
                            if self.verbose:
                                print(f"     TRUE Michel particle {true_particle_id} fragmented into {num_unique_instances} instances")
        
            for instance_id in unique_instances:
                instance_mask = instance_indices == instance_id
                instance_hits = hit_indices[instance_mask]

                if instance_hits.numel() == 0:
                    continue
                    
                if torch.any(instance_hits >= hit_store['x_raw'].size(0)) or torch.any(instance_hits < 0):
                    continue

                if torch.any(instance_hits >= probabilities.size(0)) or torch.any(instance_hits < 0):
                    continue
                    
                instance_probs = probabilities[instance_hits]
                avg_class_probs = torch.mean(instance_probs, dim=0)
                
                max_prob_class = torch.argmax(avg_class_probs)
                michel_class_idx = 3
                
                if max_prob_class == michel_class_idx:
                    michel_instances_processed += 1
                    avg_michel_prob = avg_class_probs[michel_class_idx]
                    
                    try:
                        instance_energies = hit_store['x_raw'][instance_hits, 2]
                        total_energy_raw = torch.sum(instance_energies)
                        total_energy_mev = total_energy_raw * 0.00580717
                    except (IndexError, RuntimeError) as e:
                        if self.verbose:
                            print(f"Energy calculation error for instance {instance_id}: {e}")
                        continue
                
                    is_true_michel = False
                    true_michel_purity = 0.0
                    if y_true is not None:
                        instance_gt = y_true[instance_hits]
                        true_michel_fraction = torch.sum(instance_gt == 3).float() / len(instance_hits)
                        true_michel_purity = true_michel_fraction.item()
                        is_true_michel = true_michel_fraction > 0.5
                    
                    sigma = 10.0
                    energy_diff = total_energy_mev - self.target_energy
                    gaussian_penalty = torch.exp(-0.5 * (energy_diff / sigma)**2)
                    
                    high_energy_penalty = torch.tensor(0.0, device=hit_store['x_raw'].device)
                    if total_energy_mev >= self.high_energy_threshold:
                        high_energy_penalty = self.lambda_param * avg_michel_prob * 10.0
                    
                    if y_true is not None:
                        if is_true_michel:
                            energy_loss = self.lambda_param * avg_michel_prob * (1 - gaussian_penalty)
                        else:
                            false_positive_penalty = self.lambda_param * avg_michel_prob * 5.0
                            
                            if true_michel_purity > 0 and true_michel_purity < 0.5:
                                impurity_penalty = self.lambda_param * avg_michel_prob * (1 - true_michel_purity) * 2.0
                                false_positive_penalty = false_positive_penalty + impurity_penalty
                            
                            energy_loss = false_positive_penalty
                    else:
                        energy_loss = self.lambda_param * avg_michel_prob * (1 - gaussian_penalty)
                    
                    total_instance_loss = energy_loss + high_energy_penalty
                    total_reg_loss = total_reg_loss + total_instance_loss
                    
                    if self.verbose and michel_instances_processed <= 3:
                        print(f"     Instance {instance_id}: {len(instance_hits)} hits")
                        print(f"     Class probs: MIP={avg_class_probs[0]:.3f}, HIP={avg_class_probs[1]:.3f}, "
                              f"Shower={avg_class_probs[2]:.3f}, Michel={avg_class_probs[3]:.3f}, "
                              f"Diffuse={avg_class_probs[4]:.3f}")
                        print(f"     Energy: {total_energy_mev:.2f} MeV, True Michel: {is_true_michel}, Purity: {true_michel_purity:.2f}")
                        print(f"     Energy loss: {energy_loss.item():.6f}, High-E penalty: {high_energy_penalty.item():.6f}")
            
            total_reg_loss = total_reg_loss + fragmentation_loss
        
            if self.verbose:
                print(f"   Processed {michel_instances_processed} Michel instances")
                print(f"   Fragmentation loss: {fragmentation_loss.item():.6f}")
                print(f"   Total regularization loss: {total_reg_loss.item():.6f}")
        
            return total_reg_loss
        
        except Exception as e:
            if self.verbose:
                print(f"   Regularization error: {e}")
            device = hit_store['x_raw'].device
            return torch.tensor(0.0, device=device, requires_grad=True)


def fix_batch_structure(batch):
    """Fix batch structure for all node types including sp"""
    try:
        # Get a safe device reference
        device = torch.device('cpu')
        if 'hit' in batch.node_types:
            hit_store = batch.get_node_store('hit')
            if hasattr(hit_store, 'x') and hit_store.x is not None:
                device = hit_store.x.device
        
        for node_type in batch.node_types:
            node_store = batch.get_node_store(node_type)
            
            if node_store.num_nodes == 0 or node_store.num_nodes is None:
                continue
            
            # ALWAYS ensure x exists for ALL node types
            if not hasattr(node_store, 'x') or node_store.x is None:
                if node_type == 'sp':
                    node_store.x = torch.zeros(node_store.num_nodes, 16, device=device)
                elif node_type == 'evt':
                    node_store.x = torch.zeros(node_store.num_nodes, 16, device=device)
                elif node_type == 'particle-truth':
                    node_store.x = torch.zeros(node_store.num_nodes, 8, device=device)
                elif node_type == 'hit':
                    # Hit should always have features, but just in case
                    if node_store.num_nodes > 0:
                        node_store.x = torch.zeros(node_store.num_nodes, 5, device=device)
                else:
                    # For any other node type, create minimal features
                    node_store.x = torch.zeros(node_store.num_nodes, 1, device=device)
            
            # Handle empty x tensors
            if hasattr(node_store, 'x') and node_store.x is not None and node_store.x.numel() == 0:
                if node_type == 'sp':
                    node_store.x = torch.zeros(node_store.num_nodes, 16, device=device)
                elif node_type == 'evt':
                    node_store.x = torch.zeros(node_store.num_nodes, 16, device=device)
                elif node_type == 'particle-truth':
                    node_store.x = torch.zeros(node_store.num_nodes, 8, device=device)
            
            # Fix slice_dict
            if hasattr(batch, '_slice_dict') and node_type not in batch._slice_dict:
                if hasattr(node_store, 'ptr') and node_store.ptr is not None:
                    batch._slice_dict[node_type] = {'x': node_store.ptr}
                else:
                    if batch.num_graphs > 0:
                        step_size = max(1, node_store.num_nodes // batch.num_graphs)
                        node_store.ptr = torch.arange(0, node_store.num_nodes + step_size, step_size, 
                                                     device=device, dtype=torch.long)
                        if node_store.ptr[-1] != node_store.num_nodes:
                            node_store.ptr[-1] = node_store.num_nodes
                    else:
                        node_store.ptr = torch.tensor([0, node_store.num_nodes], device=device, dtype=torch.long)
                    batch._slice_dict[node_type] = {'x': node_store.ptr}
            
            # Fix inc_dict
            if hasattr(batch, '_inc_dict') and node_type not in batch._inc_dict:
                batch._inc_dict[node_type] = {'x': torch.zeros(batch.num_graphs, device=device, dtype=torch.long)}
                    
    except Exception as e:
        print(f"Batch fixing error: {e}")
        import traceback
        traceback.print_exc()
    
    return batch


def track_fragmentation_stats(batch, frag_stats):
    """Helper to track fragmentation without logger dependency"""
    try:
        hit_store = batch.get_node_store('hit')
        
        if ('hit', 'cluster-truth', 'particle-truth') not in batch.edge_types:
            return
        if ('hit', 'cluster', 'particle') not in batch.edge_types:
            return
        if 'y_semantic' not in hit_store:
            return
        
        y_true = hit_store['y_semantic']
        
        truth_edge_type = ('hit', 'cluster-truth', 'particle-truth')
        truth_edge_index = batch[truth_edge_type].edge_index
        truth_hit_indices = truth_edge_index[0]
        truth_particle_indices = truth_edge_index[1]
        
        pred_edge_type = ('hit', 'cluster', 'particle')
        pred_edge_index = batch[pred_edge_type].edge_index
        pred_hit_indices = pred_edge_index[0]
        pred_instance_indices = pred_edge_index[1]
        
        max_hit_index = y_true.size(0) - 1
        if max_hit_index < 0:
            return
        
        valid_truth_mask = (truth_hit_indices >= 0) & (truth_hit_indices <= max_hit_index)
        if not torch.all(valid_truth_mask):
            truth_hit_indices = truth_hit_indices[valid_truth_mask]
            truth_particle_indices = truth_particle_indices[valid_truth_mask]
        
        valid_pred_mask = (pred_hit_indices >= 0) & (pred_hit_indices <= max_hit_index)
        if not torch.all(valid_pred_mask):
            pred_hit_indices = pred_hit_indices[valid_pred_mask]
            pred_instance_indices = pred_instance_indices[valid_pred_mask]
        
        if truth_hit_indices.numel() == 0 or pred_hit_indices.numel() == 0:
            return
        
        michel_mask = y_true == 3
        michel_hit_indices_set = set(torch.where(michel_mask)[0].cpu().numpy().tolist())
        
        if len(michel_hit_indices_set) == 0:
            return
        
        michel_particles = set()
        for hit_idx in michel_hit_indices_set:
            if hit_idx < 0 or hit_idx > max_hit_index:
                continue
            hit_idx_tensor = torch.tensor([hit_idx], device=truth_hit_indices.device, dtype=truth_hit_indices.dtype)
            particle_mask = truth_hit_indices == hit_idx_tensor
            if torch.any(particle_mask):
                particle_id = truth_particle_indices[particle_mask][0]
                michel_particles.add(particle_id.item())
        
        for true_particle_id in michel_particles:
            particle_mask = truth_particle_indices == true_particle_id
            true_particle_hit_indices = truth_hit_indices[particle_mask]
            
            if len(true_particle_hit_indices) == 0:
                continue
            
            instance_assignments = []
            for true_hit in true_particle_hit_indices:
                true_hit_item = true_hit.item()
                if true_hit_item < 0 or true_hit_item > max_hit_index:
                    continue
                pred_mask = pred_hit_indices == true_hit
                if torch.any(pred_mask):
                    assigned_instance = pred_instance_indices[pred_mask][0]
                    instance_assignments.append(assigned_instance.item())
            
            if len(instance_assignments) > 0:
                num_instances = len(set(instance_assignments))
                frag_stats['total_true_michel_particles'] += 1
                frag_stats['total_instances_used'] += num_instances
                frag_stats['fragmentation_ratios'].append(num_instances)
                
    except Exception as e:
        pass


def analyze_full_dataset(model, dataloader, output_dir, experiment_name):
    """Analyze complete dataset after training"""
    print("\n" + "="*60)
    print("ANALYZING FULL DATASET (all validation data)")
    print("="*60)
    
    full_analyzer = CorrectedMichelEnergyAnalyzer(overlap_threshold=0.5, verbose=False)
    
    full_frag_stats = {
        'total_true_michel_particles': 0,
        'total_instances_used': 0,
        'fragmentation_ratios': []
    }
    
    model.eval()
    model.freeze()
    
    device = next(model.parameters()).device
    
    print(f"Processing {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
            try:
                batch = batch.to(device)
                
                model.forward(batch)
                
                hit_store = batch.get_node_store('hit')
                predictions = hit_store.get('x_semantic', None)
                
                if predictions is not None:
                    full_analyzer.analyze_batch(batch, predictions)
                    track_fragmentation_stats(batch, full_frag_stats)
                
                if batch_idx % 50 == 0:
                    cleanup_memory()
                    
            except Exception as e:
                print(f"Warning: Batch {batch_idx} failed: {e}")
                continue
    
    print("\nGenerating final plots from full dataset...")
    
    energy_plot = full_analyzer.plot_energy_spectra(output_dir)
    cm_plot = full_analyzer.plot_confusion_matrix(output_dir)
    
    full_analyzer.print_diagnostics()
    
    avg_fragmentation = 1.0
    if full_frag_stats['total_true_michel_particles'] > 0:
        avg_fragmentation = (full_frag_stats['total_instances_used'] / 
                           full_frag_stats['total_true_michel_particles'])
    
    print("\nFull Dataset Fragmentation Analysis:")
    print(f"  Total true Michel particles: {full_frag_stats['total_true_michel_particles']}")
    print(f"  Average instances per particle: {avg_fragmentation:.2f}")
    print(f"  Distribution:")
    print(f"    1 instance: {sum(1 for r in full_frag_stats['fragmentation_ratios'] if r == 1)}")
    print(f"    2 instances: {sum(1 for r in full_frag_stats['fragmentation_ratios'] if r == 2)}")
    print(f"    3 instances: {sum(1 for r in full_frag_stats['fragmentation_ratios'] if r == 3)}")
    print(f"    4+ instances: {sum(1 for r in full_frag_stats['fragmentation_ratios'] if r >= 4)}")
    
    true_energies, pred_energies, validated_energies = full_analyzer.get_energy_data()
    
    full_analysis = {
        'analysis_type': 'full_validation_dataset',
        'total_batches_analyzed': len(dataloader),
        'energy_analysis': {
            'true_michel_clusters': len(true_energies),
            'predicted_michel_clusters': len(pred_energies),
            'validated_michel_clusters': len(validated_energies),
            'true_energy_mean': float(np.mean(true_energies)) if true_energies else 0,
            'predicted_energy_mean': float(np.mean(pred_energies)) if pred_energies else 0,
            'validated_energy_mean': float(np.mean(validated_energies)) if validated_energies else 0
        },
        'fragmentation_analysis': {
            'total_true_michel_particles': full_frag_stats['total_true_michel_particles'],
            'average_instances_per_particle': avg_fragmentation,
            'distribution': {
                '1_instance': sum(1 for r in full_frag_stats['fragmentation_ratios'] if r == 1),
                '2_instances': sum(1 for r in full_frag_stats['fragmentation_ratios'] if r == 2),
                '3_instances': sum(1 for r in full_frag_stats['fragmentation_ratios'] if r == 3),
                '4+_instances': sum(1 for r in full_frag_stats['fragmentation_ratios'] if r >= 4)
            }
        }
    }
    
    analysis_file = Path(output_dir) / "full_dataset_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(full_analysis, f, indent=2)
    
    print(f"\nFull dataset analysis saved: {analysis_file}")
    print("="*60)
    
    return full_analysis


class SimpleMetricsLogger:
    """Simplified metrics logging with fragmentation tracking"""

    def __init__(self, output_dir, experiment_name):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.metrics_dir = self.output_dir / "metrics"
        self.logs_dir = self.output_dir / "logs"

        for dir_path in [self.metrics_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.setup_logging()

        self.counters = {
            'total_batches': 0,
            'total_michel_pred': 0,
            'total_michel_gt': 0
        }

        self.fragmentation_stats = {
            'total_true_michel_particles': 0,
            'total_instances_used': 0,
            'fragmentation_ratios': []
        }

        self.energy_analyzer = CorrectedMichelEnergyAnalyzer(
            overlap_threshold=0.5,
            verbose=False
        )

        self.epoch_stats = {
            'epoch_numbers': [],
            'avg_fragmentation': [],
            'true_energy_mean': [],
            'pred_energy_mean': [],
            'val_energy_mean': [],
            'true_michel_count': [],
            'pred_michel_count': [],
            'fragmentation_loss': [],
            'energy_loss': []
        }
        
        # Track current epoch metrics
        self.current_epoch_frag = {
            'total_true_michel_particles': 0,
            'total_instances_used': 0,
            'fragmentation_loss_sum': 0.0,
            'energy_loss_sum': 0.0,
            'num_batches': 0
        }

        self.epoch_stats = {
            'epoch_numbers': [],
            'avg_fragmentation': [],
            'true_energy_mean': [],
            'pred_energy_mean': [],
            'val_energy_mean': [],
            'true_michel_count': [],
            'pred_michel_count': [],
            'fragmentation_loss': [],
            'energy_loss': []
        }

        self.current_epoch_frag = {
            'total_true_michel_particles': 0,
            'total_instances_used': 0,
            'fragmentation_loss_sum': 0.0,
            'energy_loss_sum': 0.0,
            'num_batches': 0
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

    def track_fragmentation(self, batch):
        """Track how fragmented true Michel particles are across predicted instances"""
        try:
            hit_store = batch.get_node_store('hit')
            
            if ('hit', 'cluster-truth', 'particle-truth') not in batch.edge_types:
                return
            if ('hit', 'cluster', 'particle') not in batch.edge_types:
                return
            if 'y_semantic' not in hit_store:
                return
            
            y_true = hit_store['y_semantic']
            
            truth_edge_type = ('hit', 'cluster-truth', 'particle-truth')
            truth_edge_index = batch[truth_edge_type].edge_index
            truth_hit_indices = truth_edge_index[0]
            truth_particle_indices = truth_edge_index[1]
            
            pred_edge_type = ('hit', 'cluster', 'particle')
            pred_edge_index = batch[pred_edge_type].edge_index
            pred_hit_indices = pred_edge_index[0]
            pred_instance_indices = pred_edge_index[1]
            
            # CRITICAL: Bounds checking
            max_hit_index = y_true.size(0) - 1
            if max_hit_index < 0:
                return
            
            valid_truth_mask = (truth_hit_indices >= 0) & (truth_hit_indices <= max_hit_index)
            if not torch.all(valid_truth_mask):
                truth_hit_indices = truth_hit_indices[valid_truth_mask]
                truth_particle_indices = truth_particle_indices[valid_truth_mask]
            
            valid_pred_mask = (pred_hit_indices >= 0) & (pred_hit_indices <= max_hit_index)
            if not torch.all(valid_pred_mask):
                pred_hit_indices = pred_hit_indices[valid_pred_mask]
                pred_instance_indices = pred_instance_indices[valid_pred_mask]
            
            if truth_hit_indices.numel() == 0 or pred_hit_indices.numel() == 0:
                return
            
            michel_mask = y_true == 3
            michel_hit_indices_set = set(torch.where(michel_mask)[0].cpu().numpy().tolist())
            
            if len(michel_hit_indices_set) == 0:
                return
            
            michel_particles = set()
            for hit_idx in michel_hit_indices_set:
                if hit_idx < 0 or hit_idx > max_hit_index:
                    continue
                hit_idx_tensor = torch.tensor([hit_idx], device=truth_hit_indices.device, dtype=truth_hit_indices.dtype)
                particle_mask = truth_hit_indices == hit_idx_tensor
                if torch.any(particle_mask):
                    particle_id = truth_particle_indices[particle_mask][0]
                    michel_particles.add(particle_id.item())
            
            for true_particle_id in michel_particles:
                particle_mask = truth_particle_indices == true_particle_id
                true_particle_hit_indices = truth_hit_indices[particle_mask]
                
                if len(true_particle_hit_indices) == 0:
                    continue
                
                instance_assignments = []
                for true_hit in true_particle_hit_indices:
                    true_hit_item = true_hit.item()
                    if true_hit_item < 0 or true_hit_item > max_hit_index:
                        continue
                    pred_mask = pred_hit_indices == true_hit
                    if torch.any(pred_mask):
                        assigned_instance = pred_instance_indices[pred_mask][0]
                        instance_assignments.append(assigned_instance.item())
                
                if len(instance_assignments) > 0:
                    num_instances = len(set(instance_assignments))
                    self.fragmentation_stats['total_true_michel_particles'] += 1
                    self.fragmentation_stats['total_instances_used'] += num_instances
                    self.fragmentation_stats['fragmentation_ratios'].append(num_instances)
                    
        except Exception as e:
            self.logger.warning(f"Fragmentation tracking failed: {e}")

    def reset_epoch_metrics(self):
        """Reset metrics at start of each epoch"""
        self.current_epoch_frag = {
            'total_true_michel_particles': 0,
            'total_instances_used': 0,
            'fragmentation_loss_sum': 0.0,
            'energy_loss_sum': 0.0,
            'num_batches': 0
        }

        self.epoch_analyzer = CorrectedMichelEnergyAnalyzer(overlap_threshold=0.5, verbose=False)

    def update_epoch_metrics(self, frag_loss, energy_loss):
        """Update running metrics for current epoch"""
        self.current_epoch_frag['fragmentation_loss_sum'] += frag_loss
        self.current_epoch_frag['energy_loss_sum'] += energy_loss
        self.current_epoch_frag['num_batches'] += 1
    
    def log_epoch_summary(self, epoch):
        """Log comprehensive epoch statistics"""
        self.logger.info("\n" + "="*70)
        self.logger.info(f"EPOCH {epoch} SUMMARY")
        self.logger.info("="*70)

        # Calculate fragmentation for this epoch
        avg_frag = 1.0
        if self.current_epoch_frag['total_true_michel_particles'] > 0:
            avg_frag = (self.current_epoch_frag['total_instances_used'] / 
                       self.current_epoch_frag['total_true_michel_particles'])

        # Get energy statistics from epoch analyzer
        true_energies, pred_energies, val_energies = self.epoch_analyzer.get_energy_data()
        
        true_energy_mean = np.mean(true_energies) if true_energies else 0.0
        pred_energy_mean = np.mean(pred_energies) if pred_energies else 0.0
        val_energy_mean = np.mean(val_energies) if val_energies else 0.0
        
        # Calculate average losses
        avg_frag_loss = 0.0
        avg_energy_loss = 0.0
        if self.current_epoch_frag['num_batches'] > 0:
            avg_frag_loss = self.current_epoch_frag['fragmentation_loss_sum'] / self.current_epoch_frag['num_batches']
            avg_energy_loss = self.current_epoch_frag['energy_loss_sum'] / self.current_epoch_frag['num_batches']
        
        # Store epoch statistics
        self.epoch_stats['epoch_numbers'].append(epoch)
        self.epoch_stats['avg_fragmentation'].append(avg_frag)
        self.epoch_stats['true_energy_mean'].append(true_energy_mean)
        self.epoch_stats['pred_energy_mean'].append(pred_energy_mean)
        self.epoch_stats['val_energy_mean'].append(val_energy_mean)
        self.epoch_stats['true_michel_count'].append(len(true_energies))
        self.epoch_stats['pred_michel_count'].append(len(pred_energies))
        self.epoch_stats['fragmentation_loss'].append(avg_frag_loss)
        self.epoch_stats['energy_loss'].append(avg_energy_loss)
        
        # Log statistics
        self.logger.info(f"\nFragmentation Metrics:")
        self.logger.info(f"  Average instances per true Michel particle: {avg_frag:.3f}")
        self.logger.info(f"  Total true Michel particles: {self.current_epoch_frag['total_true_michel_particles']}")
        self.logger.info(f"  Average fragmentation loss: {avg_frag_loss:.6f}")
        
        self.logger.info(f"\nEnergy Metrics:")
        self.logger.info(f"  True Michel energy:      {true_energy_mean:.1f} MeV (n={len(true_energies)})")
        self.logger.info(f"  Predicted Michel energy: {pred_energy_mean:.1f} MeV (n={len(pred_energies)})")
        self.logger.info(f"  Validated Michel energy: {val_energy_mean:.1f} MeV (n={len(val_energies)})")
        self.logger.info(f"  Average energy loss: {avg_energy_loss:.6f}")
        
        self.logger.info(f"\nMichel Detection:")
        self.logger.info(f"  True Michel clusters detected: {len(true_energies)}")
        self.logger.info(f"  Predicted Michel clusters: {len(pred_energies)}")
        
        # Show improvement trend if not first epoch
        if len(self.epoch_stats['epoch_numbers']) > 1:
            frag_change = avg_frag - self.epoch_stats['avg_fragmentation'][-2]
            energy_change = pred_energy_mean - self.epoch_stats['pred_energy_mean'][-2]
            
            self.logger.info(f"\nChange from previous epoch:")
            self.logger.info(f"  Fragmentation: {frag_change:+.3f} {'↓ (improving)' if frag_change < 0 else '↑ (worsening)'}")
            self.logger.info(f"  Predicted energy: {energy_change:+.1f} MeV {'↑ (improving)' if energy_change > 0 else '↓ (worsening)'}")
        
        self.logger.info("="*70 + "\n")

    def log_batch_metrics(self, epoch, batch_idx, train_loss, michel_reg_loss, total_loss, batch=None):
        """Log batch metrics and analyze energy spectra"""
        self.counters['total_batches'] += 1
        
        michel_pred = 0
        michel_gt = 0
        
        if batch is not None:
            try:
                hit_store = batch.get_node_store('hit')
                
                with torch.no_grad():
                    predictions = None
                    if 'x_semantic' in hit_store:
                        y_pred = torch.argmax(hit_store['x_semantic'], dim=1)
                        michel_pred = torch.sum(y_pred == 3).item()
                        predictions = hit_store['x_semantic']
                    
                    if 'y_semantic' in hit_store:
                        y_true = hit_store['y_semantic']
                        michel_gt = torch.sum(y_true == 3).item()
                        
                        self.counters['total_michel_pred'] += michel_pred
                        self.counters['total_michel_gt'] += michel_gt
                    
                    self.energy_analyzer.analyze_batch(batch, predictions)
                    self.track_fragmentation(batch)

                    self.epoch_analyzer.analyze_batch(batch, predictions)
                    track_fragmentation_stats(batch, self.current_epoch_frag)

                    if self.counters['total_batches'] % 50 == 0:
                        if len(self.energy_analyzer.true_clusters) > 1000:
                            self.energy_analyzer.true_clusters = self.energy_analyzer.true_clusters[-500:]
                        if len(self.energy_analyzer.predicted_clusters) > 1000:
                            self.energy_analyzer.predicted_clusters = self.energy_analyzer.predicted_clusters[-500:]
                        if len(self.energy_analyzer.validated_predicted_clusters) > 1000:
                            self.energy_analyzer.validated_predicted_clusters = self.energy_analyzer.validated_predicted_clusters[-500:]
                    
                        if len(self.energy_analyzer.all_true_labels) > 10000:
                            self.energy_analyzer.all_true_labels = self.energy_analyzer.all_true_labels[-5000:]
                            self.energy_analyzer.all_pred_labels = self.energy_analyzer.all_pred_labels[-5000:]
                        
            except Exception as e:
                self.logger.warning(f"Batch analysis failed: {e}")
        
        if batch_idx % 20 == 0:
            self.logger.info(
                f"Epoch {epoch}, Batch {batch_idx}: "
                f"Loss={train_loss:.4f}, Michel reg={michel_reg_loss:.6f}, "
                f"Total={total_loss:.4f}, Michel pred/gt={michel_pred}/{michel_gt}"
            )
    
    def create_final_plots(self):
        """Create energy spectra and confusion matrix plots"""
        try:
            energy_plot = self.energy_analyzer.plot_energy_spectra(self.output_dir)
            cm_plot = self.energy_analyzer.plot_confusion_matrix(self.output_dir)
            self.energy_analyzer.print_diagnostics()
            
            return energy_plot, cm_plot
            
        except Exception as e:
            self.logger.error(f"Plot creation failed: {e}")
            return None, None
    
    def save_final_summary(self, model_info):
        """Save final training summary with energy and fragmentation analysis"""
        true_energies, pred_energies, validated_energies = self.energy_analyzer.get_energy_data()
        
        avg_fragmentation = 1.0
        if self.fragmentation_stats['total_true_michel_particles'] > 0:
            avg_fragmentation = (self.fragmentation_stats['total_instances_used'] / 
                               self.fragmentation_stats['total_true_michel_particles'])
        
        summary = {
            'experiment_name': self.experiment_name,
            'training_completed_at': datetime.now().isoformat(),
            'model_info': model_info,
            'physics_validation': {
                'total_michel_predicted': self.counters['total_michel_pred'],
                'total_michel_ground_truth': self.counters['total_michel_gt'],
                'detection_ratio': self.counters['total_michel_pred'] / max(self.counters['total_michel_gt'], 1)
            },
            'energy_analysis': {
                'true_michel_clusters': len(true_energies),
                'predicted_michel_clusters': len(pred_energies),
                'validated_michel_clusters': len(validated_energies),
                'true_energy_stats': {
                    'mean': np.mean(true_energies) if true_energies else 0,
                    'std': np.std(true_energies) if true_energies else 0,
                    'min': np.min(true_energies) if true_energies else 0,
                    'max': np.max(true_energies) if true_energies else 0
                },
                'predicted_energy_stats': {
                    'mean': np.mean(pred_energies) if pred_energies else 0,
                    'std': np.std(pred_energies) if pred_energies else 0,
                    'min': np.min(pred_energies) if pred_energies else 0,
                    'max': np.max(pred_energies) if pred_energies else 0
                },
                'validated_energy_stats': {
                    'mean': np.mean(validated_energies) if validated_energies else 0,
                    'std': np.std(validated_energies) if validated_energies else 0,
                    'min': np.min(validated_energies) if validated_energies else 0,
                    'max': np.max(validated_energies) if validated_energies else 0
                }
            },
            'fragmentation_analysis': {
                'total_true_michel_particles': self.fragmentation_stats['total_true_michel_particles'],
                'average_instances_per_particle': avg_fragmentation,
                'perfect_clustering_would_be': 1.0,
                'fragmentation_ratios': {
                    '1_instance': sum(1 for r in self.fragmentation_stats['fragmentation_ratios'] if r == 1),
                    '2_instances': sum(1 for r in self.fragmentation_stats['fragmentation_ratios'] if r == 2),
                    '3_instances': sum(1 for r in self.fragmentation_stats['fragmentation_ratios'] if r == 3),
                    '4+_instances': sum(1 for r in self.fragmentation_stats['fragmentation_ratios'] if r >= 4)
                }
            },
            'improvements_applied': [
                'Updated target energy to 40 MeV (matches actual data)',
                'REMOVED all hard thresholds - pure max probability approach',
                'Added anti-fragmentation regularization (Option 1)',
                'Added impurity penalty for low-purity clusters',
                'Ground truth validation penalty',
                'Comprehensive SP node fixes',
                'Full dataset analysis after training',
                'Three-way energy spectrum analysis',
                'Confusion matrix generation',
                'Overlap-based validation filtering',
                'Reduced sigma to 10.0 for tighter energy constraint',
                'Strong penalty for clusters ≥100 MeV',
                'Fragmentation metrics tracking',
                'Comprehensive bounds checking to prevent CUDA errors'
            ],
            'epoch_progression': {
                'epochs': self.epoch_stats['epoch_numbers'],
                'fragmentation': self.epoch_stats['avg_fragmentation'],
                'true_energy_mean': self.epoch_stats['true_energy_mean'],
                'predicted_energy_mean': self.epoch_stats['pred_energy_mean'],
                'validated_energy_mean': self.epoch_stats['val_energy_mean'],
                'true_michel_count': self.epoch_stats['true_michel_count'],
                'predicted_michel_count': self.epoch_stats['pred_michel_count'],
                'fragmentation_loss': self.epoch_stats['fragmentation_loss'],
                'energy_loss': self.epoch_stats['energy_loss']
            },
        }
        
        summary_file = self.output_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary saved: {summary_file}")
        return summary


def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/nugraph/NG2-paper.gnn.h5')
    parser.add_argument('--output-dir', type=str, default='./improved_training_outputs')
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--michel-reg-lambda', type=float, default=1.0)
    parser.add_argument('--target-energy', type=float, default=40.0)
    parser.add_argument('--baseline', action='store_true', 
                       help='Run baseline training without energy regularization')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None)
    return parser.parse_args()


def train(args):
    """Main training function with improved Michel physics and energy analysis"""
    try:
        experiment_dir = Path(args.output_dir) / args.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        metrics_logger = SimpleMetricsLogger(experiment_dir, args.experiment_name)

        metrics_logger.logger.info(f"Training mode: {'BASELINE (no regularization)' if args.baseline else 'REGULARIZED WITH ANTI-FRAGMENTATION'}")
        metrics_logger.logger.info(f"Improved Michel training: {args.experiment_name}")
        metrics_logger.logger.info(f"Target energy: {args.target_energy} MeV")
        if not args.baseline:
            metrics_logger.logger.info(f"Regularization strength: {args.michel_reg_lambda}")
        else:
            metrics_logger.logger.info("Regularization: DISABLED for baseline")

        import nugraph
        from nugraph.data import H5DataModule
        from nugraph.models.nugraph3.decoders.spacepoint import SpacepointDecoder
        import nugraph.models.nugraph3.decoders
        nugraph.models.nugraph3.decoders.SpacepointDecoder = SpacepointDecoder
        from nugraph.models.nugraph3.nugraph3 import NuGraph3

        metrics_logger.logger.info(f'Loading data: {args.data_path}')
        nudata = H5DataModule(args.data_path, batch_size=args.batch_size, model=NuGraph3)
        original_transform = NuGraph3.transform(nudata.planes)

        def fixed_transform(data):
            data = original_transform(data)
            data = fix_batch_structure(data)
            return data

        nudata.transform = fixed_transform

        original_train_dataloader = nudata.train_dataloader
        original_val_dataloader = nudata.val_dataloader



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

        metrics_logger.logger.info('Creating NuGraph3 model')
        if args.resume_from_checkpoint:
            metrics_logger.logger.info(f'Loading model from checkpoint: {args.resume_from_checkpoint}')
            model = NuGraph3.load_from_checkpoint(
                args.resume_from_checkpoint,
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
                instance_head=True,
                spacepoint_head=False,
                use_checkpointing=True,
                lr=args.learning_rate
            )
        else:
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
                instance_head=True,
                spacepoint_head=False,
                use_checkpointing=True,
                lr=args.learning_rate
            )

        def fix_sp_node_features(batch):
            try:
                sp_store = batch.get_node_store('sp')
                if not hasattr(sp_store, 'x') or sp_store.x is None or sp_store.x.size(1) == 0:
                    device = batch.get_node_store('hit').x.device
                    sp_store.x = torch.zeros(sp_store.num_nodes, 16, device=device)
            except Exception as e:
                print(f"Warning: Could not fix sp node features: {e}")
            return batch

        original_forward = model.forward

        def fixed_forward(data, stage=None):
            data = fix_sp_node_features(data)
            return original_forward(data, stage)

        model.forward = fixed_forward

        original_validation_step = model.validation_step

        def fixed_validation_step(batch, batch_idx):
            try:
                sp_store = batch.get_node_store('sp')

                if not hasattr(sp_store, 'x') or sp_store.x is None or sp_store.x.size(1) == 0:
                    device = batch.get_node_store('hit').x.device
                    sp_store.x = torch.zeros(sp_store.num_nodes, 16, device=device)

                if hasattr(batch, '_slice_dict') and 'sp' not in batch._slice_dict:
                    batch._slice_dict['sp'] = {'x': sp_store.ptr if hasattr(sp_store, 'ptr') else torch.tensor([0, sp_store.num_nodes], device=device)}

                if hasattr(batch, '_inc_dict') and 'sp' not in batch._inc_dict:
                    batch._inc_dict['sp'] = {'x': torch.zeros(batch.num_graphs, device=device, dtype=torch.long)}

                return original_validation_step(batch, batch_idx)

            except Exception as e:
                print(f"Validation step error: {e}")
                return torch.tensor(0.0, device=batch.get_node_store('hit').x.device)

        model.validation_step = fixed_validation_step

        if torch.cuda.is_available():
            model = model.cuda()
            metrics_logger.logger.info("Model on GPU")

        michel_reg = None
        if not args.baseline:
            michel_reg = ImprovedMichelRegularizer(
                lambda_param=args.michel_reg_lambda,
                target_energy=args.target_energy,
                verbose=False
            )

        batch_losses = []
        michel_losses = []
        original_training_step = model.training_step

        def enhanced_training_step(batch, batch_idx):
            nonlocal batch_losses, michel_losses
            try:
                if torch.cuda.is_available():
                    batch = batch.to('cuda')

                loss = original_training_step(batch, batch_idx)
                reg_loss_val = 0.0
                
                if michel_reg is not None:
                    reg_loss = michel_reg(batch)
                    total_loss = loss + reg_loss
                    reg_loss_val = reg_loss.item() if hasattr(reg_loss, 'item') else float(reg_loss)
                else:
                    reg_loss = torch.tensor(0.0, device=loss.device)
                    total_loss = loss

                model.log('michel_reg_loss', reg_loss, batch_size=getattr(batch, 'num_graphs', 1))
                model.log('total_loss', total_loss, batch_size=getattr(batch, 'num_graphs', 1))

                loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
                reg_loss_val = reg_loss.item() if hasattr(reg_loss, 'item') else float(reg_loss)
                total_loss_val = total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)

                batch_losses.append(loss_val)
                michel_losses.append(reg_loss_val)

                current_epoch = getattr(model.trainer, 'current_epoch', 0) if hasattr(model, 'trainer') else 0

                metrics_logger.log_batch_metrics(
                    current_epoch, batch_idx, loss_val, reg_loss_val, total_loss_val, batch
                )

                metrics_logger.update_epoch_metrics(frag_loss=0.0, energy_loss=reg_loss_val)
                cleanup_batch_memory(batch)

                if batch_idx % 10 == 0:
                    cleanup_memory()

                if len(batch_losses) > 100:
                    batch_losses = batch_losses[-50:]
                if len(michel_losses) > 100:
                    michel_losses = michel_losses[-50:]

                return total_loss

            except Exception as e:
                metrics_logger.logger.error(f"Training step failed: {e}")
                cleanup_memory()
                return torch.tensor(1.0, device=next(model.parameters()).device, requires_grad=True)

        model.training_step = enhanced_training_step

        checkpoint_dir = experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch}-{loss/train:.4f}',
            save_top_k=3,
            monitor='loss/train',
            mode='min',
            save_last=True
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[checkpoint_callback, MemoryCleanupCallback(metrics_logger)],
            logger=False,
            enable_progress_bar=True,
            deterministic=True
        )

        metrics_logger.logger.info(f"Starting training for {args.epochs} epochs")
        start_time = time.time()

        if args.resume_from_checkpoint:
            trainer.fit(model, nudata, ckpt_path=args.resume_from_checkpoint)
        else:
            trainer.fit(model, nudata)

        total_time = time.time() - start_time
        metrics_logger.logger.info(f"Training completed in {total_time/3600:.2f} hours")

        metrics_logger.logger.info("Creating plots from training batches...")
        energy_plot, cm_plot = metrics_logger.create_final_plots()

        model_info = {
            'model_type': f'NuGraph3_{"Baseline" if args.baseline else "MaxProb_AntiFrag"}',
            'training_mode': 'baseline' if args.baseline else 'regularized_with_anti_fragmentation',
            'epochs_trained': args.epochs,
            'target_energy_mev': args.target_energy,
            'regularization_lambda': 0.0 if args.baseline else args.michel_reg_lambda,
            'total_training_time_hours': total_time / 3600,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'approach': 'none' if args.baseline else 'max_probability_no_thresholds_anti_fragmentation',
            'sigma': 10.0 if not args.baseline else 'not_applicable',
            'high_energy_penalty': 100.0 if not args.baseline else 'not_applicable'
        }

        final_summary = metrics_logger.save_final_summary(model_info)

        metrics_logger.logger.info("TRAINING SUMMARY (from training batches):")
        metrics_logger.logger.info("=" * 50)
        physics_val = final_summary.get('physics_validation', {})
        energy_analysis = final_summary.get('energy_analysis', {})

        metrics_logger.logger.info(f"Total Michel predicted: {physics_val.get('total_michel_predicted', 0)}")
        metrics_logger.logger.info(f"Total Michel ground truth: {physics_val.get('total_michel_gt', 0)}")
        metrics_logger.logger.info(f"Detection ratio: {physics_val.get('detection_ratio', 0):.3f}")

        metrics_logger.logger.info("\nEnergy Analysis:")
        metrics_logger.logger.info(f"True Michel clusters: {energy_analysis.get('true_michel_clusters', 0)}")
        metrics_logger.logger.info(f"Predicted Michel clusters: {energy_analysis.get('predicted_michel_clusters', 0)}")
        metrics_logger.logger.info(f"Validated Michel clusters: {energy_analysis.get('validated_michel_clusters', 0)}")

        true_stats = energy_analysis.get('true_energy_stats', {})
        pred_stats = energy_analysis.get('predicted_energy_stats', {})
        val_stats = energy_analysis.get('validated_energy_stats', {})

        if true_stats.get('mean', 0) > 0:
            metrics_logger.logger.info(f"True Michel energy: {true_stats['mean']:.1f}±{true_stats['std']:.1f} MeV")
        if pred_stats.get('mean', 0) > 0:
            metrics_logger.logger.info(f"Predicted Michel energy: {pred_stats['mean']:.1f}±{pred_stats['std']:.1f} MeV")
        if val_stats.get('mean', 0) > 0:
            metrics_logger.logger.info(f"Validated Michel energy: {val_stats['mean']:.1f}±{val_stats['std']:.1f} MeV")

        frag_analysis = final_summary.get('fragmentation_analysis', {})
        metrics_logger.logger.info("\nFragmentation Analysis (from training):")
        metrics_logger.logger.info(f"Average instances per true Michel particle: {frag_analysis.get('average_instances_per_particle', 0):.2f}")
        metrics_logger.logger.info(f"Perfect clustering would be: 1.0")

        frag_ratios = frag_analysis.get('fragmentation_ratios', {})
        metrics_logger.logger.info("Fragmentation distribution:")
        metrics_logger.logger.info(f"  1 instance (perfect): {frag_ratios.get('1_instance', 0)}")
        metrics_logger.logger.info(f"  2 instances: {frag_ratios.get('2_instances', 0)}")
        metrics_logger.logger.info(f"  3 instances: {frag_ratios.get('3_instances', 0)}")
        metrics_logger.logger.info(f"  4+ instances: {frag_ratios.get('4+_instances', 0)}")

        metrics_logger.logger.info("=" * 50)

        if energy_plot:
            metrics_logger.logger.info(f"Energy spectra plot: {energy_plot}")
        if cm_plot:
            metrics_logger.logger.info(f"Confusion matrix plot: {cm_plot}")

        metrics_logger.logger.info(f"All outputs saved to: {experiment_dir}")

        # NEW: Analyze full validation dataset
        metrics_logger.logger.info("\nPerforming full dataset analysis...")
        try:
            full_analysis = analyze_full_dataset(
                model,
                nudata.val_dataloader(),
                experiment_dir,
                args.experiment_name
            )

            metrics_logger.logger.info("Full dataset analysis completed!")
        except Exception as e:
            metrics_logger.logger.error(f"Full dataset analysis failed: {e}")
            import traceback
            traceback.print_exc()

        return model, trainer

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    print("Michel Electron Training with Energy Analysis and Anti-Fragmentation")
    print("=" * 60)

    try:
        args = configure()

        if args.baseline:
            print("BASELINE MODE - No Energy Regularization")
            print("Fragmentation will be tracked but NOT penalized")
        else:
            print("REGULARIZED MODE - Max Probability + Anti-Fragmentation")
            print("Improvements:")
            print("   - Target energy: 40 MeV")
            print("   - Pure max probability approach")
            print("   - Anti-fragmentation penalty ")
            print("   - Impurity penalty for low-purity clusters")
            print("   - Ground truth validation penalties")
            print("   - Fragmentation metrics tracking")
            print("   - Full dataset analysis after training")
            print("   - Comprehensive bounds checking")
            print("   - Three-way energy spectrum analysis")
            print("   - Confusion matrix generation")
            print("   - Strong penalty for clusters ≥100 MeV")

        print("=" * 60)
        print(f"Experiment: {args.experiment_name}")
        print(f"Target energy: {args.target_energy} MeV")
        if not args.baseline:
            print(f"Regularization: λ={args.michel_reg_lambda}")
        else:
            print("Regularization: DISABLED")
        print(f"Epochs: {args.epochs}")
        print("=" * 60)

        model, trainer = train(args)

        if model is not None and trainer is not None:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Check results in: {args.output_dir}/{args.experiment_name}/")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
