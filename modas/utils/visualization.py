"""
Visualization Utilities for MODAS

Plotting functions for:
- Filter visualizations (V1, A1)
- Training curves
- Sparsity distributions
- ATL activations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Optional, Tuple, Dict
from pathlib import Path


def plot_filters(
    filters: torch.Tensor,
    n_cols: int = 16,
    figsize: Optional[Tuple[int, int]] = None,
    title: str = "Learned Filters",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot learned filters in a grid.
    
    Args:
        filters: Filters (n_filters, H, W) or (n_filters, H, W, C)
        n_cols: Number of columns in grid
        figsize: Figure size (width, height)
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
    
    Returns:
        Matplotlib figure
    """
    filters = filters.detach().cpu().numpy()
    n_filters = filters.shape[0]
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (n_cols * 1.5, n_rows * 1.5)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax = axes[i, j]
            
            if idx < n_filters:
                filt = filters[idx]
                
                # Handle grayscale vs color
                if filt.ndim == 3 and filt.shape[-1] == 3:
                    # Normalize for display
                    filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
                    ax.imshow(filt)
                else:
                    ax.imshow(filt, cmap='gray')
                
            ax.axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_dictionary(
    model,
    n_cols: int = 16,
    title: str = "Dictionary",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize model dictionary/filters.
    
    Args:
        model: Model with visualize_dictionary() method
        n_cols: Number of columns
        title: Plot title
        save_path: Save path
        show: Whether to display
    
    Returns:
        Matplotlib figure
    """
    if hasattr(model, 'visualize_dictionary'):
        filters = model.visualize_dictionary()
    elif hasattr(model, 'dictionary'):
        # V1: reshape (n_bases, patch_size*patch_size*C) -> (n_bases, H, W, C)
        if hasattr(model, 'patch_size'):
            ps = model.patch_size
            nc = model.n_channels if hasattr(model, 'n_channels') else 1
            filters = model.dictionary.reshape(-1, nc, ps, ps).permute(0, 2, 3, 1)
        else:
            filters = model.dictionary
    else:
        raise ValueError("Model has no dictionary to visualize")
    
    return plot_filters(filters, n_cols, title=title, save_path=save_path, show=show)


def plot_training_curves(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 4),
    title: str = "Training Progress",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot training curves.
    
    Args:
        history: Dictionary of metric_name -> list of values
        figsize: Figure size
        title: Plot title
        save_path: Save path
        show: Whether to display
    
    Returns:
        Matplotlib figure
    """
    n_metrics = len(history)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, history.items()):
        ax.plot(values)
        ax.set_xlabel('Step')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_sparsity_histogram(
    codes: torch.Tensor,
    threshold: float = 0.01,
    figsize: Tuple[int, int] = (10, 4),
    title: str = "Activation Distribution",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot histogram of activations with sparsity statistics.
    
    Args:
        codes: Activation codes (batch, n_units) or (n_units,)
        threshold: Threshold for "active"
        figsize: Figure size
        title: Plot title
        save_path: Save path
        show: Whether to display
    
    Returns:
        Matplotlib figure
    """
    codes = codes.detach().cpu().numpy().flatten()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Full distribution
    ax1.hist(codes, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
    ax1.axvline(-threshold, color='r', linestyle='--')
    ax1.set_xlabel('Activation')
    ax1.set_ylabel('Count')
    ax1.set_title('Full Distribution')
    ax1.legend()
    
    # Log-scale distribution
    codes_abs = np.abs(codes)
    ax2.hist(codes_abs[codes_abs > 1e-6], bins=50, alpha=0.7, edgecolor='black')
    ax2.set_yscale('log')
    ax2.axvline(threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
    ax2.set_xlabel('|Activation|')
    ax2.set_ylabel('Count (log)')
    ax2.set_title('Magnitude Distribution (log scale)')
    ax2.legend()
    
    # Statistics
    sparsity = (np.abs(codes) <= threshold).mean()
    mean_active = np.abs(codes[np.abs(codes) > threshold]).mean() if np.any(np.abs(codes) > threshold) else 0
    
    fig.suptitle(f"{title}\nSparsity: {sparsity:.1%}, Mean active: {mean_active:.3f}", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_discrimination_results(
    matched_sims: List[float],
    mismatched_sims: List[float],
    figsize: Tuple[int, int] = (10, 4),
    title: str = "Discrimination Analysis",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot discrimination results (matched vs mismatched similarities).
    
    Args:
        matched_sims: Matched pair similarities
        mismatched_sims: Mismatched pair similarities
        figsize: Figure size
        title: Plot title
        save_path: Save path
        show: Whether to display
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histograms
    ax1.hist(matched_sims, bins=30, alpha=0.7, label='Matched', color='green')
    ax1.hist(mismatched_sims, bins=30, alpha=0.7, label='Mismatched', color='red')
    ax1.axvline(np.mean(matched_sims), color='darkgreen', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(mismatched_sims), color='darkred', linestyle='--', linewidth=2)
    ax1.set_xlabel('Similarity')
    ax1.set_ylabel('Count')
    ax1.set_title('Similarity Distributions')
    ax1.legend()
    
    # Box plot
    ax2.boxplot([matched_sims, mismatched_sims], labels=['Matched', 'Mismatched'])
    ax2.set_ylabel('Similarity')
    ax2.set_title('Box Plot')
    
    # Compute discrimination
    disc = np.mean(matched_sims) - np.mean(mismatched_sims)
    
    # Determine status
    if disc > 0.2:
        status = "EXCELLENT"
        color = "green"
    elif disc > 0.15:
        status = "GOOD"
        color = "blue"
    elif disc > 0.1:
        status = "MARGINAL"
        color = "orange"
    else:
        status = "FAIL"
        color = "red"
    
    fig.suptitle(f"{title}\nDiscrimination: {disc:.3f} ({status})", 
                 fontsize=12, color=color)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_capacity_usage(
    usage_count: torch.Tensor,
    threshold: float = 0.01,
    figsize: Tuple[int, int] = (10, 4),
    title: str = "Capacity Usage",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot capacity usage distribution.
    
    FROM MBM LESSON: Monitor capacity to catch collapse early.
    
    Args:
        usage_count: Usage counts per unit
        threshold: Threshold for "used"
        figsize: Figure size
        title: Plot title
        save_path: Save path
        show: Whether to display
    
    Returns:
        Matplotlib figure
    """
    usage = usage_count.detach().cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Usage distribution
    ax1.bar(range(len(usage)), np.sort(usage)[::-1], alpha=0.7)
    ax1.axhline(threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
    ax1.set_xlabel('Unit (sorted)')
    ax1.set_ylabel('Usage count')
    ax1.set_title('Usage Distribution (sorted)')
    ax1.legend()
    
    # Log usage
    ax2.bar(range(len(usage)), np.sort(usage)[::-1], alpha=0.7)
    ax2.set_yscale('log')
    ax2.axhline(threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
    ax2.set_xlabel('Unit (sorted)')
    ax2.set_ylabel('Usage count (log)')
    ax2.set_title('Usage Distribution (log scale)')
    ax2.legend()
    
    # Effective capacity
    capacity = (usage > threshold).mean()
    
    fig.suptitle(f"{title}\nEffective Capacity: {capacity:.1%}", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_spectrogram_filters(
    filters: torch.Tensor,
    n_cols: int = 8,
    figsize: Optional[Tuple[int, int]] = None,
    title: str = "A1 Temporal Filters",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot A1 spectrogram filters.
    
    Args:
        filters: Filters (n_filters, n_freq, n_time)
        n_cols: Number of columns
        figsize: Figure size
        title: Plot title
        save_path: Save path
        show: Whether to display
    
    Returns:
        Matplotlib figure
    """
    filters = filters.detach().cpu().numpy()
    n_filters = filters.shape[0]
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (n_cols * 2, n_rows * 1.5)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax = axes[i, j]
            
            if idx < n_filters:
                filt = filters[idx]
                ax.imshow(filt, aspect='auto', cmap='RdBu_r', origin='lower')
            
            ax.axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def create_summary_figure(
    v1_filters: Optional[torch.Tensor] = None,
    a1_filters: Optional[torch.Tensor] = None,
    training_history: Optional[Dict[str, List[float]]] = None,
    matched_sims: Optional[List[float]] = None,
    mismatched_sims: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create comprehensive summary figure.
    
    Args:
        v1_filters: V1 filters
        a1_filters: A1 filters
        training_history: Training curves
        matched_sims: Matched similarities
        mismatched_sims: Mismatched similarities
        figsize: Figure size
        save_path: Save path
        show: Whether to display
    
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig)
    
    # V1 filters
    if v1_filters is not None:
        ax1 = fig.add_subplot(gs[0, 0])
        v1_np = v1_filters.detach().cpu().numpy()
        # Show subset
        n_show = min(64, len(v1_np))
        grid_size = int(np.ceil(np.sqrt(n_show)))
        combined = np.zeros((grid_size * v1_np.shape[1], grid_size * v1_np.shape[2]))
        for i in range(n_show):
            row, col = i // grid_size, i % grid_size
            filt = v1_np[i]
            if filt.ndim == 3:
                filt = filt.mean(axis=-1)
            combined[row*filt.shape[0]:(row+1)*filt.shape[0],
                    col*filt.shape[1]:(col+1)*filt.shape[1]] = filt
        ax1.imshow(combined, cmap='gray')
        ax1.set_title('V1 Filters')
        ax1.axis('off')
    
    # A1 filters
    if a1_filters is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        a1_np = a1_filters.detach().cpu().numpy()
        n_show = min(16, len(a1_np))
        for i in range(n_show):
            ax2.plot(a1_np[i].flatten(), alpha=0.5)
        ax2.set_title('A1 Temporal Filters')
        ax2.set_xlabel('Time')
    
    # Training curves
    if training_history is not None:
        ax3 = fig.add_subplot(gs[1, :])
        for name, values in training_history.items():
            ax3.plot(values, label=name)
        ax3.legend()
        ax3.set_xlabel('Step')
        ax3.set_title('Training Progress')
        ax3.grid(True, alpha=0.3)
    
    # Discrimination
    if matched_sims is not None and mismatched_sims is not None:
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.hist(matched_sims, bins=20, alpha=0.7, label='Matched', color='green')
        ax4.hist(mismatched_sims, bins=20, alpha=0.7, label='Mismatched', color='red')
        ax4.legend()
        ax4.set_title('Discrimination')
        
        ax5 = fig.add_subplot(gs[2, 1])
        disc = np.mean(matched_sims) - np.mean(mismatched_sims)
        ax5.text(0.5, 0.5, f'Discrimination: {disc:.3f}', 
                fontsize=24, ha='center', va='center',
                transform=ax5.transAxes)
        if disc > 0.15:
            ax5.set_facecolor('#d4edda')  # Green
        elif disc > 0.1:
            ax5.set_facecolor('#fff3cd')  # Yellow
        else:
            ax5.set_facecolor('#f8d7da')  # Red
        ax5.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig
