"""
V1 Sparse Coding Validation

Validates V1 module for:
1. Gabor filter emergence
2. Sparsity
3. Discrimination (different images → different codes)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from modas.modules.v1_sparse_coding import V1SparseCoding
from modas.utils.metrics import (
    compute_sparsity,
    measure_gabor_similarity,
    create_gabor_bank,
    compute_cross_similarity,
)
from modas.utils.visualization import (
    plot_filters,
    plot_sparsity_histogram,
    plot_capacity_usage,
)


class V1Validator:
    """
    Validator for V1 sparse coding module.
    
    Success criteria:
    - Gabor score > 0.6 (filters look like oriented edges)
    - Sparsity > 0.85 (< 15% neurons active)
    - Cross-similarity < 0.5 (different images → different codes)
    """
    
    GABOR_THRESHOLD = 0.6
    SPARSITY_THRESHOLD = 0.85
    CROSS_SIM_THRESHOLD = 0.5
    
    def __init__(
        self,
        model: V1SparseCoding,
        device: str = 'cpu',
    ):
        self.model = model.to(device)
        self.device = device
        self.gabor_bank = create_gabor_bank(size=model.patch_size)
    
    def validate(
        self,
        test_images: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Full validation suite.
        
        Args:
            test_images: Test images (batch, C, H, W) or patches (batch, dim)
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        test_images = test_images.to(self.device)
        
        # Get codes
        with torch.no_grad():
            if test_images.dim() == 4:
                # Full images
                codes = self.model.forward(test_images)
            else:
                # Patches
                codes = self.model.lca_inference(test_images)
        
        results = {}
        
        # 1. Gabor similarity
        filters = self.model.visualize_dictionary()
        results['gabor_score'] = measure_gabor_similarity(filters, self.gabor_bank)
        
        # 2. Sparsity
        results['sparsity'] = compute_sparsity(codes)
        
        # 3. Cross-similarity
        codes_norm = F.normalize(codes, p=2, dim=1)
        sim_matrix = codes_norm @ codes_norm.T
        mask = ~torch.eye(len(codes), dtype=torch.bool, device=self.device)
        results['cross_similarity'] = sim_matrix[mask].mean().item()
        
        # 4. Effective capacity
        results['effective_capacity'] = self.model.get_effective_capacity()
        
        # 5. Reconstruction error
        if test_images.dim() == 2:
            recon = codes @ self.model.dictionary
            results['reconstruction_mse'] = F.mse_loss(recon, test_images).item()
        
        return results
    
    def check_pass(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if validation passed.
        
        Returns:
            (passed, status_message)
        """
        gabor_pass = metrics.get('gabor_score', 0) > self.GABOR_THRESHOLD
        sparsity_pass = metrics.get('sparsity', 0) > self.SPARSITY_THRESHOLD
        cross_sim_pass = metrics.get('cross_similarity', 1) < self.CROSS_SIM_THRESHOLD
        
        checks = {
            'Gabor': (gabor_pass, metrics.get('gabor_score', 0), f'>{self.GABOR_THRESHOLD}'),
            'Sparsity': (sparsity_pass, metrics.get('sparsity', 0), f'>{self.SPARSITY_THRESHOLD}'),
            'CrossSim': (cross_sim_pass, metrics.get('cross_similarity', 1), f'<{self.CROSS_SIM_THRESHOLD}'),
        }
        
        status_parts = []
        for name, (passed, value, threshold) in checks.items():
            symbol = '✓' if passed else '✗'
            status_parts.append(f"{name}: {value:.3f} {symbol} ({threshold})")
        
        all_passed = gabor_pass and sparsity_pass and cross_sim_pass
        status = '\n'.join(status_parts)
        
        return all_passed, status
    
    def visualize(
        self,
        test_images: torch.Tensor,
        save_dir: str,
    ):
        """Generate validation visualizations."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Filters
        filters = self.model.visualize_dictionary()
        plot_filters(
            filters,
            title='V1 Learned Filters',
            save_path=str(save_dir / 'v1_filters.png'),
            show=False,
        )
        
        # Sparsity histogram
        test_images = test_images.to(self.device)
        with torch.no_grad():
            if test_images.dim() == 4:
                codes = self.model.forward(test_images)
            else:
                codes = self.model.lca_inference(test_images)
        
        plot_sparsity_histogram(
            codes,
            title='V1 Activation Distribution',
            save_path=str(save_dir / 'v1_sparsity.png'),
            show=False,
        )
        
        # Capacity usage
        plot_capacity_usage(
            self.model.usage_count,
            title='V1 Capacity Usage',
            save_path=str(save_dir / 'v1_capacity.png'),
            show=False,
        )


def validate_v1(
    model_path: str,
    test_data: torch.Tensor,
    device: str = 'cpu',
    save_dir: Optional[str] = None,
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate V1 model from checkpoint.
    
    Args:
        model_path: Path to V1 checkpoint
        test_data: Test images or patches
        device: Device
        save_dir: Directory for visualizations
    
    Returns:
        (passed, metrics)
    """
    # Load model
    model = V1SparseCoding()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    # Validate
    validator = V1Validator(model, device)
    metrics = validator.validate(test_data)
    passed, status = validator.check_pass(metrics)
    
    print("=" * 50)
    print("V1 VALIDATION RESULTS")
    print("=" * 50)
    print(status)
    print("=" * 50)
    print(f"RESULT: {'PASS ✓' if passed else 'FAIL ✗'}")
    print("=" * 50)
    
    if save_dir:
        validator.visualize(test_data, save_dir)
    
    return passed, metrics
