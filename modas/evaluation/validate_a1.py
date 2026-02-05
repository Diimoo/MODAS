"""
A1 Sparse Coding Validation

Validates A1 module for:
1. Temporal filter structure
2. Sparsity
3. Discrimination
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

from modas.modules.a1_sparse_coding import A1SparseCoding
from modas.utils.metrics import compute_sparsity, measure_temporal_structure
from modas.utils.visualization import (
    plot_spectrogram_filters,
    plot_sparsity_histogram,
    plot_capacity_usage,
)


class A1Validator:
    """
    Validator for A1 sparse coding module.
    
    Success criteria:
    - Temporal structure score > 0.3
    - Sparsity > 0.85
    - Cross-similarity < 0.5
    """
    
    TEMPORAL_THRESHOLD = 0.3
    SPARSITY_THRESHOLD = 0.85
    CROSS_SIM_THRESHOLD = 0.5
    
    def __init__(
        self,
        model: A1SparseCoding,
        device: str = 'cpu',
    ):
        self.model = model.to(device)
        self.device = device
    
    def validate(
        self,
        test_audio: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Full validation suite.
        
        Args:
            test_audio: Test audio segments (batch, segment_length)
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        test_audio = test_audio.to(self.device)
        
        # Get codes
        with torch.no_grad():
            codes = self.model.forward(test_audio)
        
        results = {}
        
        # 1. Temporal structure
        filters = self.model.visualize_dictionary()
        results['temporal_score'] = measure_temporal_structure(filters)
        
        # 2. Sparsity
        results['sparsity'] = compute_sparsity(codes)
        
        # 3. Cross-similarity
        codes_norm = F.normalize(codes, p=2, dim=1)
        sim_matrix = codes_norm @ codes_norm.T
        mask = ~torch.eye(len(codes), dtype=torch.bool, device=self.device)
        results['cross_similarity'] = sim_matrix[mask].mean().item()
        
        # 4. Effective capacity
        results['effective_capacity'] = self.model.get_effective_capacity()
        
        return results
    
    def check_pass(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """Check if validation passed."""
        temporal_pass = metrics.get('temporal_score', 0) > self.TEMPORAL_THRESHOLD
        sparsity_pass = metrics.get('sparsity', 0) > self.SPARSITY_THRESHOLD
        cross_sim_pass = metrics.get('cross_similarity', 1) < self.CROSS_SIM_THRESHOLD
        
        checks = {
            'Temporal': (temporal_pass, metrics.get('temporal_score', 0), f'>{self.TEMPORAL_THRESHOLD}'),
            'Sparsity': (sparsity_pass, metrics.get('sparsity', 0), f'>{self.SPARSITY_THRESHOLD}'),
            'CrossSim': (cross_sim_pass, metrics.get('cross_similarity', 1), f'<{self.CROSS_SIM_THRESHOLD}'),
        }
        
        status_parts = []
        for name, (passed, value, threshold) in checks.items():
            symbol = '✓' if passed else '✗'
            status_parts.append(f"{name}: {value:.3f} {symbol} ({threshold})")
        
        all_passed = temporal_pass and sparsity_pass and cross_sim_pass
        
        return all_passed, '\n'.join(status_parts)
    
    def visualize(
        self,
        test_audio: torch.Tensor,
        save_dir: str,
    ):
        """Generate validation visualizations."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Filters
        filters = self.model.visualize_dictionary()
        plot_spectrogram_filters(
            filters,
            title='A1 Temporal Filters',
            save_path=str(save_dir / 'a1_filters.png'),
            show=False,
        )
        
        # Sparsity
        test_audio = test_audio.to(self.device)
        with torch.no_grad():
            codes = self.model.forward(test_audio)
        
        plot_sparsity_histogram(
            codes,
            title='A1 Activation Distribution',
            save_path=str(save_dir / 'a1_sparsity.png'),
            show=False,
        )
        
        # Capacity
        plot_capacity_usage(
            self.model.usage_count,
            title='A1 Capacity Usage',
            save_path=str(save_dir / 'a1_capacity.png'),
            show=False,
        )


def validate_a1(
    model_path: str,
    test_data: torch.Tensor,
    device: str = 'cpu',
    save_dir: Optional[str] = None,
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate A1 model from checkpoint.
    
    Args:
        model_path: Path to A1 checkpoint
        test_data: Test audio segments
        device: Device
        save_dir: Directory for visualizations
    
    Returns:
        (passed, metrics)
    """
    model = A1SparseCoding()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    validator = A1Validator(model, device)
    metrics = validator.validate(test_data)
    passed, status = validator.check_pass(metrics)
    
    print("=" * 50)
    print("A1 VALIDATION RESULTS")
    print("=" * 50)
    print(status)
    print("=" * 50)
    print(f"RESULT: {'PASS ✓' if passed else 'FAIL ✗'}")
    print("=" * 50)
    
    if save_dir:
        validator.visualize(test_data, save_dir)
    
    return passed, metrics
