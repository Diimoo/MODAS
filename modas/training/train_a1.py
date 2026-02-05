"""
A1 Sparse Coding Training

Train A1 module on audio using temporal sparse coding.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm

from modas.modules.a1_sparse_coding import A1SparseCoding
from modas.utils.metrics import compute_sparsity, measure_temporal_structure
from modas.utils.checkpointing import CheckpointManager
from modas.utils.visualization import plot_spectrogram_filters, plot_training_curves


class A1Trainer:
    """
    Trainer for A1 sparse coding module.
    
    Args:
        model: A1SparseCoding model
        device: Training device
        checkpoint_dir: Directory for checkpoints
        log_interval: Logging interval
        val_interval: Validation interval
    """
    
    def __init__(
        self,
        model: A1SparseCoding,
        device: str = 'cpu',
        checkpoint_dir: str = 'experiments/phase2_a1/checkpoints',
        log_interval: int = 100,
        val_interval: int = 10,
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.val_interval = val_interval
        
        self.history = {
            'mse': [],
            'sparsity': [],
            'temporal_score': [],
            'capacity': [],
        }
        
        self.ckpt_manager = CheckpointManager(
            str(self.checkpoint_dir),
            max_checkpoints=5,
            metric_name='temporal_score',
            higher_is_better=True,
        )
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_mse = 0
        total_sparsity = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, segments in enumerate(pbar):
            segments = segments.to(self.device)
            
            batch_mse = 0
            batch_codes = []
            
            for segment in segments:
                # Compute spectrogram
                spec = self.model.compute_spectrogram(segment.unsqueeze(0)).squeeze(0)
                spec_flat = spec.flatten()
                
                # Normalize
                spec_flat = spec_flat - spec_flat.mean()
                std = spec_flat.std()
                if std > 1e-6:
                    spec_flat = spec_flat / std
                
                # LCA inference
                code = self.model.lca_inference(spec_flat.unsqueeze(0)).squeeze(0)
                
                # Learn
                mse = self.model.learn(spec_flat, code)
                batch_mse += mse
                batch_codes.append(code)
            
            batch_mse /= len(segments)
            codes = torch.stack(batch_codes)
            sparsity = compute_sparsity(codes)
            
            total_mse += batch_mse
            total_sparsity += sparsity
            n_batches += 1
            
            if (batch_idx + 1) % self.log_interval == 0:
                pbar.set_postfix({
                    'MSE': f'{batch_mse:.4f}',
                    'Sparsity': f'{sparsity:.2%}',
                })
        
        avg_mse = total_mse / n_batches
        avg_sparsity = total_sparsity / n_batches
        
        self.history['mse'].append(avg_mse)
        self.history['sparsity'].append(avg_sparsity)
        
        return {'mse': avg_mse, 'sparsity': avg_sparsity}
    
    def validate(self, test_segments: torch.Tensor) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        test_segments = test_segments.to(self.device)
        
        with torch.no_grad():
            codes = self.model.forward(test_segments)
        
        sparsity = compute_sparsity(codes)
        
        # Temporal structure
        filters = self.model.visualize_dictionary()
        temporal_score = measure_temporal_structure(filters)
        
        capacity = self.model.get_effective_capacity()
        
        # Cross-similarity
        codes_norm = F.normalize(codes, p=2, dim=1)
        sim_matrix = codes_norm @ codes_norm.T
        mask = ~torch.eye(len(codes), dtype=torch.bool, device=self.device)
        cross_sim = sim_matrix[mask].mean().item()
        
        self.history['temporal_score'].append(temporal_score)
        self.history['capacity'].append(capacity)
        
        return {
            'sparsity': sparsity,
            'temporal_score': temporal_score,
            'capacity': capacity,
            'cross_similarity': cross_sim,
        }
    
    def check_convergence(self, metrics: Dict[str, float]) -> tuple:
        """Check convergence criteria."""
        sparsity = metrics.get('sparsity', 0)
        temporal = metrics.get('temporal_score', 0)
        cross_sim = metrics.get('cross_similarity', 1)
        
        checks = {
            'sparsity': sparsity > 0.85,
            'temporal': temporal > 0.3,
            'cross_sim': cross_sim < 0.5,
        }
        
        status_parts = [f"{k}: {'✓' if v else '✗'}" for k, v in checks.items()]
        return all(checks.values()), ', '.join(status_parts)
    
    def train(
        self,
        train_dataloader,
        test_segments: torch.Tensor,
        n_epochs: int = 100,
        early_stop: bool = True,
    ) -> Dict[str, float]:
        """Full training loop."""
        print(f"Starting A1 training for {n_epochs} epochs")
        
        best_metrics = None
        
        for epoch in range(1, n_epochs + 1):
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            if epoch % self.val_interval == 0 or epoch == 1:
                val_metrics = self.validate(test_segments)
                
                print(f"\nEpoch {epoch} Validation:")
                print(f"  MSE: {train_metrics['mse']:.4f}")
                print(f"  Sparsity: {val_metrics['sparsity']:.2%}")
                print(f"  Temporal Score: {val_metrics['temporal_score']:.3f}")
                print(f"  Capacity: {val_metrics['capacity']:.2%}")
                
                converged, status = self.check_convergence(val_metrics)
                print(f"  Status: {status}")
                
                self.ckpt_manager.save(self.model, None, epoch, {**train_metrics, **val_metrics})
                
                if converged:
                    print(f"\n✓ A1 CONVERGED at epoch {epoch}!")
                    best_metrics = val_metrics
                    if early_stop:
                        break
                
                best_metrics = val_metrics
        
        final_path = self.checkpoint_dir / 'a1_final.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'metrics': best_metrics,
        }, final_path)
        
        return best_metrics or {}
    
    def visualize(self, save_dir: Optional[str] = None):
        """Visualize filters and training curves."""
        if save_dir is None:
            save_dir = str(self.checkpoint_dir / 'figures')
        
        filters = self.model.visualize_dictionary()
        plot_spectrogram_filters(
            filters,
            title='A1 Temporal Filters',
            save_path=f'{save_dir}/a1_filters.png',
            show=False,
        )
        
        if len(self.history['mse']) > 0:
            plot_training_curves(
                self.history,
                title='A1 Training Progress',
                save_path=f'{save_dir}/a1_training.png',
                show=False,
            )


def train_a1(
    audio_dir: str,
    n_epochs: int = 100,
    batch_size: int = 64,
    n_bases: int = 128,
    segment_length: int = 3200,
    device: str = 'cpu',
    checkpoint_dir: str = 'experiments/phase2_a1/checkpoints',
    **kwargs
) -> A1SparseCoding:
    """
    Train A1 sparse coding model.
    
    Args:
        audio_dir: Directory containing audio files
        n_epochs: Number of epochs
        batch_size: Batch size
        n_bases: Number of bases
        segment_length: Audio segment length
        device: Device
        checkpoint_dir: Checkpoint directory
    
    Returns:
        Trained A1SparseCoding model
    """
    from modas.data.audio_dataset import create_audio_dataloader
    
    model = A1SparseCoding(n_bases=n_bases, segment_length=segment_length, **kwargs)
    
    dataloader = create_audio_dataloader(
        audio_dir,
        batch_size=batch_size,
        segment_length=segment_length,
    )
    
    # Create test segments
    test_segments = []
    for batch in dataloader:
        test_segments.append(batch)
        if len(test_segments) * batch_size >= 200:
            break
    test_segments = torch.cat(test_segments)[:200]
    
    trainer = A1Trainer(model, device, checkpoint_dir)
    trainer.train(dataloader, test_segments, n_epochs)
    trainer.visualize()
    
    return model
