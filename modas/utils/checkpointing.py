"""
Checkpointing Utilities for MODAS

Save and load model checkpoints with training state.
"""

import torch
import os
from typing import Dict, Any, Optional
from pathlib import Path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    path: str,
    metrics: Optional[Dict[str, float]] = None,
    extra_state: Optional[Dict[str, Any]] = None,
):
    """
    Save model checkpoint with training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        epoch: Current epoch number
        path: Save path
        metrics: Dictionary of metrics to save
        extra_state: Additional state to save
    """
    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if extra_state is not None:
        checkpoint['extra_state'] = extra_state
    
    # Save custom state if available
    if hasattr(model, 'state_dict_custom'):
        checkpoint['custom_state'] = model.state_dict_custom()
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load state into
        path: Checkpoint path
        optimizer: Optimizer to load state into (optional)
        device: Device to load to
    
    Returns:
        Dictionary with epoch, metrics, and extra_state
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load custom state if available
    if 'custom_state' in checkpoint and hasattr(model, 'load_state_dict_custom'):
        model.load_state_dict_custom(checkpoint['custom_state'])
    
    print(f"Checkpoint loaded: {path}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'extra_state': checkpoint.get('extra_state', {}),
    }


def save_model(
    model: torch.nn.Module,
    path: str,
    include_config: bool = True,
):
    """
    Save model weights only (for inference).
    
    Args:
        model: PyTorch model
        path: Save path
        include_config: Include model configuration
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if include_config:
        # Save relevant config attributes
        config = {}
        for attr in ['n_bases', 'n_prototypes', 'feature_dim', 'patch_size',
                     'segment_length', 'embedding_dim', 'vocab_size']:
            if hasattr(model, attr):
                config[attr] = getattr(model, attr)
        save_dict['config'] = config
    
    # Save custom state
    if hasattr(model, 'state_dict_custom'):
        save_dict['custom_state'] = model.state_dict_custom()
    
    torch.save(save_dict, path)
    print(f"Model saved: {path}")


def load_model(
    model: torch.nn.Module,
    path: str,
    device: str = 'cpu',
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model weights.
    
    Args:
        model: PyTorch model
        path: Model path
        device: Device to load to
        strict: Strict state dict loading
    
    Returns:
        Configuration dictionary
    """
    save_dict = torch.load(path, map_location=device)
    
    model.load_state_dict(save_dict['model_state_dict'], strict=strict)
    
    if 'custom_state' in save_dict and hasattr(model, 'load_state_dict_custom'):
        model.load_state_dict_custom(save_dict['custom_state'])
    
    print(f"Model loaded: {path}")
    
    return save_dict.get('config', {})


def get_best_checkpoint(
    checkpoint_dir: str,
    metric_name: str = 'discrimination',
    higher_is_better: bool = True,
) -> Optional[str]:
    """
    Find best checkpoint based on metric.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        metric_name: Metric to compare
        higher_is_better: Whether higher metric is better
    
    Returns:
        Path to best checkpoint, or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob('*.pt')) + list(checkpoint_dir.glob('*.pth'))
    
    if len(checkpoints) == 0:
        return None
    
    best_path = None
    best_metric = float('-inf') if higher_is_better else float('inf')
    
    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            metrics = ckpt.get('metrics', {})
            
            if metric_name in metrics:
                metric_val = metrics[metric_name]
                
                if higher_is_better and metric_val > best_metric:
                    best_metric = metric_val
                    best_path = str(ckpt_path)
                elif not higher_is_better and metric_val < best_metric:
                    best_metric = metric_val
                    best_path = str(ckpt_path)
        except Exception:
            continue
    
    return best_path


def checkpoint_exists(path: str) -> bool:
    """Check if checkpoint exists."""
    return os.path.exists(path)


class CheckpointManager:
    """
    Manages checkpoint saving with automatic cleanup.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        metric_name: str = 'discrimination',
        higher_is_better: bool = True,
    ):
        """
        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum checkpoints to keep
            metric_name: Metric for comparison
            higher_is_better: Whether higher is better
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.checkpoints = []
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict[str, float],
    ):
        """Save checkpoint and manage cleanup."""
        # Create filename
        metric_val = metrics.get(self.metric_name, 0)
        filename = f"checkpoint_epoch{epoch:04d}_disc{metric_val:.4f}.pt"
        path = self.checkpoint_dir / filename
        
        save_checkpoint(model, optimizer, epoch, str(path), metrics)
        
        self.checkpoints.append({
            'path': str(path),
            'epoch': epoch,
            'metric': metric_val,
        })
        
        # Cleanup old checkpoints
        self._cleanup()
    
    def _cleanup(self):
        """Remove old checkpoints, keeping best ones."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by metric
        self.checkpoints.sort(
            key=lambda x: x['metric'],
            reverse=self.higher_is_better
        )
        
        # Keep top max_checkpoints
        to_remove = self.checkpoints[self.max_checkpoints:]
        self.checkpoints = self.checkpoints[:self.max_checkpoints]
        
        for ckpt in to_remove:
            try:
                os.remove(ckpt['path'])
                print(f"Removed old checkpoint: {ckpt['path']}")
            except Exception:
                pass
    
    def get_best(self) -> Optional[str]:
        """Get path to best checkpoint."""
        if len(self.checkpoints) == 0:
            return None
        
        self.checkpoints.sort(
            key=lambda x: x['metric'],
            reverse=self.higher_is_better
        )
        return self.checkpoints[0]['path']
