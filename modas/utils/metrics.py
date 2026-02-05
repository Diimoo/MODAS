"""
Evaluation Metrics for MODAS

Includes metrics for:
- Sparsity measurement
- Cross-similarity / discrimination
- Gabor filter emergence detection
- Capacity monitoring
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Callable


def compute_sparsity(
    codes: torch.Tensor,
    threshold: float = 0.01
) -> float:
    """
    Compute activation sparsity (fraction of active units).
    
    Args:
        codes: Activation codes (batch, n_units) or (n_units,)
        threshold: Activation threshold for "active"
    
    Returns:
        Sparsity ratio (0 = all active, 1 = all inactive)
    """
    return (codes.abs() <= threshold).float().mean().item()


def compute_lifetime_sparsity(
    codes: torch.Tensor,
    threshold: float = 0.01
) -> torch.Tensor:
    """
    Compute per-unit lifetime sparsity (how often each unit is active).
    
    Args:
        codes: Activation codes (batch, n_units)
        threshold: Activation threshold
    
    Returns:
        Per-unit sparsity (n_units,)
    """
    return (codes.abs() > threshold).float().mean(dim=0)


def compute_cross_similarity(
    model: Callable,
    inputs: List[torch.Tensor],
    device: str = 'cpu'
) -> float:
    """
    Compute average pairwise similarity of codes for different inputs.
    
    Lower is better (different inputs should have different codes).
    
    Args:
        model: Model with forward() method returning codes
        inputs: List of input tensors
        device: Device for computation
    
    Returns:
        Mean pairwise cosine similarity
    """
    codes = []
    for inp in inputs:
        inp = inp.to(device)
        code = model.forward(inp)
        if code.dim() > 1:
            code = code.flatten()
        codes.append(F.normalize(code, p=2, dim=0))
    
    if len(codes) < 2:
        return 0.0
    
    # Compute pairwise similarities
    sims = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            sim = F.cosine_similarity(
                codes[i].unsqueeze(0),
                codes[j].unsqueeze(0)
            ).item()
            sims.append(sim)
    
    return np.mean(sims)


def compute_discrimination(
    matched_sims: List[float],
    mismatched_sims: List[float]
) -> float:
    """
    Compute discrimination score.
    
    discrimination = mean(matched) - mean(mismatched)
    
    Target: > 0.15 (bio-plausible), > 0.2 (excellent)
    
    Args:
        matched_sims: Similarities for matched pairs
        mismatched_sims: Similarities for mismatched pairs
    
    Returns:
        Discrimination score
    """
    if len(matched_sims) == 0 or len(mismatched_sims) == 0:
        return 0.0
    
    return np.mean(matched_sims) - np.mean(mismatched_sims)


def compute_discrimination_full(
    model,
    test_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    get_similarity: Callable,
    device: str = 'cpu'
) -> Tuple[float, float, float]:
    """
    Compute full discrimination metrics.
    
    Args:
        model: Model for encoding
        test_pairs: List of (input1, input2) matched pairs
        get_similarity: Function(model, inp1, inp2) -> similarity
        device: Device
    
    Returns:
        Tuple of (discrimination, mean_matched, mean_mismatched)
    """
    matched_sims = []
    mismatched_sims = []
    
    # Matched similarities
    for inp1, inp2 in test_pairs:
        inp1, inp2 = inp1.to(device), inp2.to(device)
        sim = get_similarity(model, inp1, inp2)
        matched_sims.append(sim)
    
    # Mismatched similarities (all pairs)
    for i, (inp1, _) in enumerate(test_pairs):
        for j, (_, inp2) in enumerate(test_pairs):
            if i != j:
                inp1, inp2 = inp1.to(device), inp2.to(device)
                sim = get_similarity(model, inp1, inp2)
                mismatched_sims.append(sim)
    
    mean_matched = np.mean(matched_sims)
    mean_mismatched = np.mean(mismatched_sims)
    disc = mean_matched - mean_mismatched
    
    return disc, mean_matched, mean_mismatched


def create_gabor_bank(
    size: int = 16,
    orientations: List[float] = None,
    frequencies: List[float] = None,
    sigmas: List[float] = None,
) -> torch.Tensor:
    """
    Create bank of Gabor filters for comparison.
    
    Args:
        size: Filter size (size x size)
        orientations: List of orientations in degrees
        frequencies: List of spatial frequencies
        sigmas: List of Gaussian envelope sigmas
    
    Returns:
        Gabor filter bank (n_filters, size, size)
    """
    if orientations is None:
        orientations = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    if frequencies is None:
        frequencies = [0.1, 0.15, 0.2, 0.25, 0.3]
    if sigmas is None:
        sigmas = [2.0, 3.0, 4.0]
    
    gabors = []
    
    for theta_deg in orientations:
        for freq in frequencies:
            for sigma in sigmas:
                gabor = _create_gabor(size, theta_deg, freq, sigma)
                gabors.append(gabor)
    
    return torch.stack(gabors)


def _create_gabor(
    size: int,
    theta_deg: float,
    frequency: float,
    sigma: float
) -> torch.Tensor:
    """Create single Gabor filter."""
    theta = np.radians(theta_deg)
    
    # Create coordinate grid
    x = np.linspace(-size/2, size/2, size)
    y = np.linspace(-size/2, size/2, size)
    X, Y = np.meshgrid(x, y)
    
    # Rotate coordinates
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
    
    # Gaussian envelope
    gaussian = np.exp(-(X_rot**2 + Y_rot**2) / (2 * sigma**2))
    
    # Sinusoidal carrier
    sinusoid = np.cos(2 * np.pi * frequency * X_rot)
    
    # Gabor = Gaussian × Sinusoid
    gabor = gaussian * sinusoid
    
    # Normalize
    gabor = gabor / (np.linalg.norm(gabor) + 1e-8)
    
    return torch.tensor(gabor, dtype=torch.float32)


def measure_gabor_similarity(
    filters: torch.Tensor,
    gabor_bank: Optional[torch.Tensor] = None,
    size: int = 16,
) -> float:
    """
    Measure how similar learned filters are to Gabor filters.
    
    Args:
        filters: Learned filters (n_filters, H, W) or (n_filters, H, W, C)
        gabor_bank: Reference Gabor bank (optional, will create if None)
        size: Filter size (if creating Gabor bank)
    
    Returns:
        Mean maximum similarity to Gabor filters (0-1)
    """
    # Handle color filters (average across channels)
    if filters.dim() == 4:
        filters = filters.mean(dim=-1)  # Average over channels
    
    # Flatten filters
    n_filters = filters.shape[0]
    filters_flat = filters.reshape(n_filters, -1)
    
    # Normalize
    filters_flat = F.normalize(filters_flat, p=2, dim=1)
    
    # Create Gabor bank if not provided
    if gabor_bank is None:
        filter_size = filters.shape[1]
        gabor_bank = create_gabor_bank(size=filter_size)
    
    # Flatten Gabor bank
    n_gabors = gabor_bank.shape[0]
    gabors_flat = gabor_bank.reshape(n_gabors, -1)
    gabors_flat = F.normalize(gabors_flat, p=2, dim=1)
    
    # Compute similarities
    # For each learned filter, find max similarity to any Gabor
    max_sims = []
    for i in range(n_filters):
        sims = filters_flat[i] @ gabors_flat.T  # (n_gabors,)
        max_sim = sims.abs().max().item()  # abs because Gabor can be inverted
        max_sims.append(max_sim)
    
    return np.mean(max_sims)


def measure_temporal_structure(
    filters: torch.Tensor,
    n_freq: int = None,
    n_time: int = None,
) -> float:
    """
    Measure temporal structure in audio filters.
    
    Checks for frequency selectivity and time localization.
    
    Args:
        filters: Learned filters (n_filters, n_freq, n_time) or flattened
        n_freq: Number of frequency bins (if flattened)
        n_time: Number of time frames (if flattened)
    
    Returns:
        Temporal structure score (0-1)
    """
    if filters.dim() == 2 and n_freq is not None and n_time is not None:
        filters = filters.reshape(-1, n_freq, n_time)
    
    n_filters = filters.shape[0]
    
    scores = []
    for i in range(n_filters):
        filt = filters[i]
        
        # Frequency selectivity: variance across frequencies
        freq_var = filt.var(dim=0).mean().item()
        
        # Time localization: variance across time
        time_var = filt.var(dim=1).mean().item()
        
        # Good filters have high variance (localized)
        score = (freq_var + time_var) / 2
        scores.append(score)
    
    # Normalize by max possible variance
    max_score = max(scores) if scores else 1.0
    return np.mean(scores) / (max_score + 1e-8)


def random_baseline_test(
    model,
    test_data,
    compute_metric: Callable,
    n_trials: int = 5,
) -> Tuple[float, float, float]:
    """
    Compare learned model to random baseline.
    
    FROM CHPL LESSON: Always compare to random to catch artifacts.
    
    Args:
        model: Learned model
        test_data: Test data
        compute_metric: Function(model, data) -> metric
        n_trials: Number of random trials
    
    Returns:
        Tuple of (learned_metric, random_metric, improvement)
    """
    import copy
    
    # Learned model metric
    learned_metric = compute_metric(model, test_data)
    
    # Random baseline (average over trials)
    random_metrics = []
    for _ in range(n_trials):
        random_model = copy.deepcopy(model)
        
        # Reset to random weights
        if hasattr(random_model, 'prototypes'):
            random_model.prototypes = F.normalize(
                torch.randn_like(random_model.prototypes), dim=1
            )
        if hasattr(random_model, 'dictionary'):
            random_model.dictionary = F.normalize(
                torch.randn_like(random_model.dictionary), dim=1
            )
        if hasattr(random_model, 'embeddings'):
            random_model.embeddings = F.normalize(
                torch.randn_like(random_model.embeddings), dim=1
            )
        
        random_metric = compute_metric(random_model, test_data)
        random_metrics.append(random_metric)
    
    random_metric = np.mean(random_metrics)
    improvement = learned_metric - random_metric
    
    return learned_metric, random_metric, improvement


def compute_effective_capacity(
    usage_count: torch.Tensor,
    threshold: float = 0.01
) -> float:
    """
    Compute effective capacity (fraction of used units).
    
    FROM MBM LESSON: Monitor capacity to catch collapse early.
    
    Args:
        usage_count: Usage count per unit
        threshold: Minimum usage to count as "used"
    
    Returns:
        Effective capacity (0-1)
    """
    return (usage_count > threshold).float().mean().item()
