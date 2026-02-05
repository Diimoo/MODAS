"""
V1 Sparse Coding Training

Train V1 module on natural images using LCA sparse coding.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Callable
from tqdm import tqdm

from modas.modules.v1_sparse_coding import V1SparseCoding
from modas.utils.metrics import compute_sparsity, measure_gabor_similarity, create_gabor_bank
from modas.utils.checkpointing import save_checkpoint, CheckpointManager
from modas.utils.visualization import plot_filters, plot_training_curves


class V1Trainer:
    """
    Trainer for V1 sparse coding module.
    
    Args:
        model: V1SparseCoding model
        device: Training device
        checkpoint_dir: Directory for checkpoints
        log_interval: Logging interval (steps)
        val_interval: Validation interval (epochs)
    """
    
    def __init__(
        self,
        model: V1SparseCoding,
        device: str = 'cpu',
        checkpoint_dir: str = 'experiments/phase1_v1/checkpoints',
        log_interval: int = 100,
        val_interval: int = 10,
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.val_interval = val_interval
        
        # Training history
        self.history = {
            'mse': [],
            'sparsity': [],
            'gabor_score': [],
            'capacity': [],
        }
        
        # Checkpoint manager
        self.ckpt_manager = CheckpointManager(
            str(self.checkpoint_dir),
            max_checkpoints=5,
            metric_name='gabor_score',
            higher_is_better=True,
        )
        
        # Gabor bank for validation
        self.gabor_bank = create_gabor_bank(size=model.patch_size)
    
    def train_epoch(
        self,
        dataloader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_mse = 0
        total_sparsity = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, patches in enumerate(pbar):
            patches = patches.to(self.device)
            
            # Process each patch in batch
            batch_mse = 0
            batch_codes = []
            
            for patch in patches:
                # LCA inference
                code = self.model.lca_inference(patch.unsqueeze(0)).squeeze(0)
                
                # Learn
                mse = self.model.learn(patch, code)
                batch_mse += mse
                batch_codes.append(code)
            
            batch_mse /= len(patches)
            codes = torch.stack(batch_codes)
            sparsity = compute_sparsity(codes)
            
            total_mse += batch_mse
            total_sparsity += sparsity
            n_batches += 1
            
            # Log
            if (batch_idx + 1) % self.log_interval == 0:
                pbar.set_postfix({
                    'MSE': f'{batch_mse:.4f}',
                    'Sparsity': f'{sparsity:.2%}',
                })
        
        avg_mse = total_mse / n_batches
        avg_sparsity = total_sparsity / n_batches
        
        self.history['mse'].append(avg_mse)
        self.history['sparsity'].append(avg_sparsity)
        
        return {
            'mse': avg_mse,
            'sparsity': avg_sparsity,
        }
    
    def validate(self, test_patches: torch.Tensor) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        test_patches = test_patches.to(self.device)
        
        # Get codes for test patches
        with torch.no_grad():
            codes = self.model.lca_inference(test_patches)
        
        # Sparsity
        sparsity = compute_sparsity(codes)
        
        # Gabor similarity
        filters = self.model.visualize_dictionary()
        gabor_score = measure_gabor_similarity(filters, self.gabor_bank)
        
        # Effective capacity
        capacity = self.model.get_effective_capacity()
        
        # Cross-similarity (different patches should have different codes)
        codes_norm = F.normalize(codes, p=2, dim=1)
        sim_matrix = codes_norm @ codes_norm.T
        # Exclude diagonal
        mask = ~torch.eye(len(codes), dtype=torch.bool, device=self.device)
        cross_sim = sim_matrix[mask].mean().item()
        
        self.history['gabor_score'].append(gabor_score)
        self.history['capacity'].append(capacity)
        
        return {
            'sparsity': sparsity,
            'gabor_score': gabor_score,
            'capacity': capacity,
            'cross_similarity': cross_sim,
        }
    
    def check_convergence(self, metrics: Dict[str, float]) -> tuple:
        """
        Check if training has converged.
        
        Success criteria:
        - Gabor score > 0.6
        - Sparsity > 0.85 (i.e., < 15% active)
        - Cross-similarity < 0.5
        
        Returns:
            (converged: bool, status: str)
        """
        gabor = metrics.get('gabor_score', 0)
        sparsity = metrics.get('sparsity', 0)
        cross_sim = metrics.get('cross_similarity', 1)
        
        checks = {
            'gabor': gabor > 0.6,
            'sparsity': sparsity > 0.85,
            'cross_sim': cross_sim < 0.5,
        }
        
        status_parts = []
        for name, passed in checks.items():
            symbol = '✓' if passed else '✗'
            status_parts.append(f"{name}: {symbol}")
        
        status = ', '.join(status_parts)
        converged = all(checks.values())
        
        return converged, status
    
    def train(
        self,
        train_dataloader,
        test_patches: torch.Tensor,
        n_epochs: int = 100,
        early_stop: bool = True,
    ) -> Dict[str, float]:
        """
        Full training loop.
        
        Args:
            train_dataloader: Training data loader
            test_patches: Test patches for validation
            n_epochs: Maximum epochs
            early_stop: Stop early if converged
        
        Returns:
            Final metrics
        """
        print(f"Starting V1 training for {n_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.n_bases} bases, {self.model.patch_size}x{self.model.patch_size} patches")
        
        best_metrics = None
        
        for epoch in range(1, n_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Validate
            if epoch % self.val_interval == 0 or epoch == 1:
                val_metrics = self.validate(test_patches)
                
                print(f"\nEpoch {epoch} Validation:")
                print(f"  MSE: {train_metrics['mse']:.4f}")
                print(f"  Sparsity: {val_metrics['sparsity']:.2%}")
                print(f"  Gabor Score: {val_metrics['gabor_score']:.3f}")
                print(f"  Capacity: {val_metrics['capacity']:.2%}")
                print(f"  Cross-Sim: {val_metrics['cross_similarity']:.3f}")
                
                # Check convergence
                converged, status = self.check_convergence(val_metrics)
                print(f"  Status: {status}")
                
                # Save checkpoint
                self.ckpt_manager.save(
                    self.model, None, epoch,
                    {**train_metrics, **val_metrics}
                )
                
                if converged:
                    print(f"\n✓ V1 CONVERGED at epoch {epoch}!")
                    best_metrics = val_metrics
                    if early_stop:
                        break
                
                best_metrics = val_metrics
        
        # Save final model
        final_path = self.checkpoint_dir / 'v1_final.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'metrics': best_metrics,
        }, final_path)
        print(f"Final model saved: {final_path}")
        
        return best_metrics or {}
    
    def visualize(self, save_dir: Optional[str] = None):
        """Visualize learned filters and training curves."""
        if save_dir is None:
            save_dir = str(self.checkpoint_dir / 'figures')
        
        # Plot filters
        filters = self.model.visualize_dictionary()
        plot_filters(
            filters,
            title='V1 Learned Filters',
            save_path=f'{save_dir}/v1_filters.png',
            show=False,
        )
        
        # Plot training curves
        if len(self.history['mse']) > 0:
            plot_training_curves(
                self.history,
                title='V1 Training Progress',
                save_path=f'{save_dir}/v1_training.png',
                show=False,
            )


def train_v1(
    image_dir: str,
    n_epochs: int = 100,
    batch_size: int = 64,
    n_bases: int = 128,
    patch_size: int = 16,
    device: str = 'cpu',
    checkpoint_dir: str = 'experiments/phase1_v1/checkpoints',
    **kwargs
) -> V1SparseCoding:
    """
    Train V1 sparse coding model.
    
    Args:
        image_dir: Directory containing training images
        n_epochs: Number of epochs
        batch_size: Batch size
        n_bases: Number of dictionary bases
        patch_size: Patch size
        device: Training device
        checkpoint_dir: Checkpoint directory
        **kwargs: Additional model arguments
    
    Returns:
        Trained V1SparseCoding model
    """
    from modas.data.image_dataset import create_image_dataloader
    
    # Create model
    model = V1SparseCoding(
        n_bases=n_bases,
        patch_size=patch_size,
        **kwargs
    )
    
    # Create data loader
    dataloader = create_image_dataloader(
        image_dir,
        batch_size=batch_size,
        patch_size=patch_size,
    )
    
    # Create test patches
    test_patches = []
    for batch in dataloader:
        test_patches.append(batch)
        if len(test_patches) * batch_size >= 500:
            break
    test_patches = torch.cat(test_patches)[:500]
    
    # Train
    trainer = V1Trainer(model, device, checkpoint_dir)
    trainer.train(dataloader, test_patches, n_epochs)
    trainer.visualize()
    
    return model
