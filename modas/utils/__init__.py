"""
MODAS Utility Functions

- metrics: Evaluation metrics (sparsity, discrimination, Gabor similarity)
- checkpointing: Model save/load utilities
- visualization: Plotting functions for filters and training curves
"""

from modas.utils.metrics import (
    compute_sparsity,
    compute_cross_similarity,
    compute_discrimination,
    measure_gabor_similarity,
    create_gabor_bank,
)
from modas.utils.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    save_model,
    load_model,
)
from modas.utils.visualization import (
    plot_filters,
    plot_training_curves,
    plot_sparsity_histogram,
    visualize_dictionary,
)

__all__ = [
    # Metrics
    "compute_sparsity",
    "compute_cross_similarity",
    "compute_discrimination",
    "measure_gabor_similarity",
    "create_gabor_bank",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "save_model",
    "load_model",
    # Visualization
    "plot_filters",
    "plot_training_curves",
    "plot_sparsity_histogram",
    "visualize_dictionary",
]
