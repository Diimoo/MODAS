"""
ATL Semantic Hub Validation

THE CRITICAL TEST for MODAS.

Validates ATL binding for:
1. Discrimination (matched vs mismatched pairs)
2. Compositional generalization (swap tests)
3. Novel combination handling
4. Random baseline comparison
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import copy

from modas.modules.atl_semantic_hub import ATLSemanticHub
from modas.modules.v1_sparse_coding import V1SparseCoding
from modas.modules.language_encoder import LanguageEncoder
from modas.modules.a1_sparse_coding import A1SparseCoding
from modas.utils.visualization import plot_discrimination_results, plot_capacity_usage


class ATLValidator:
    """
    Validator for ATL semantic hub.
    
    THE critical test of MODAS.
    
    Success criteria (from spec):
    - EXCELLENT: discrimination > 0.2
    - GOOD: discrimination > 0.15 (bio-plausible target)
    - MARGINAL: discrimination > 0.1
    - FAIL: discrimination <= 0.1
    """
    
    EXCELLENT_THRESHOLD = 0.2
    GOOD_THRESHOLD = 0.15
    MARGINAL_THRESHOLD = 0.1
    
    def __init__(
        self,
        atl: ATLSemanticHub,
        v1: V1SparseCoding,
        language: LanguageEncoder,
        a1: Optional[A1SparseCoding] = None,
        device: str = 'cpu',
    ):
        self.atl = atl.to(device)
        self.v1 = v1.to(device)
        self.language = language
        self.a1 = a1.to(device) if a1 is not None else None
        self.device = device
        
        # Freeze feature extractors
        self.v1.eval()
        if self.a1:
            self.a1.eval()
    
    def get_features(
        self,
        image: torch.Tensor,
        text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from image and text."""
        with torch.no_grad():
            vis_code = self.v1.forward(image.to(self.device))
            if vis_code.dim() > 1:
                vis_code = vis_code.squeeze(0)
        
        lang_emb = self.language.forward(text)
        if lang_emb.device != self.device:
            lang_emb = lang_emb.to(self.device)
        
        return vis_code, lang_emb
    
    def compute_discrimination(
        self,
        test_samples: List[Tuple[torch.Tensor, str]],
    ) -> Dict[str, float]:
        """
        Compute discrimination score.
        
        discrimination = mean(matched_sim) - mean(mismatched_sim)
        
        Args:
            test_samples: List of (image, text) pairs
        
        Returns:
            Dictionary with discrimination metrics
        """
        self.atl.eval()
        
        # Get features and ATL activations for all samples
        features = []
        for image, text in test_samples:
            vis_code, lang_emb = self.get_features(image, text)
            vis_act = self.atl.forward(vis_code)
            lang_act = self.atl.forward(lang_emb)
            features.append((vis_act, lang_act, text))
        
        # Compute matched similarities
        matched_sims = []
        for vis_act, lang_act, _ in features:
            sim = F.cosine_similarity(
                vis_act.unsqueeze(0),
                lang_act.unsqueeze(0)
            ).item()
            matched_sims.append(sim)
        
        # Compute mismatched similarities
        mismatched_sims = []
        for i, (vis_act, _, _) in enumerate(features):
            for j, (_, lang_act, _) in enumerate(features):
                if i != j:
                    sim = F.cosine_similarity(
                        vis_act.unsqueeze(0),
                        lang_act.unsqueeze(0)
                    ).item()
                    mismatched_sims.append(sim)
        
        discrimination = np.mean(matched_sims) - np.mean(mismatched_sims)
        
        return {
            'discrimination': discrimination,
            'mean_matched': np.mean(matched_sims),
            'std_matched': np.std(matched_sims),
            'mean_mismatched': np.mean(mismatched_sims),
            'std_mismatched': np.std(mismatched_sims),
            'matched_sims': matched_sims,
            'mismatched_sims': mismatched_sims,
        }
    
    def test_swap_generalization(
        self,
        test_samples: List[Tuple[torch.Tensor, str]],
    ) -> float:
        """
        Test compositional swap generalization.
        
        For samples that differ by one attribute (e.g., "red circle" vs "blue circle"),
        check if the model correctly distinguishes them.
        """
        # This requires samples with structured labels
        # For now, return -1 if not applicable
        
        swap_scores = []
        
        for i in range(len(test_samples)):
            for j in range(i + 1, len(test_samples)):
                img_i, text_i = test_samples[i]
                img_j, text_j = test_samples[j]
                
                # Check if texts share some words (potential swap pair)
                words_i = set(text_i.lower().split())
                words_j = set(text_j.lower().split())
                shared = words_i & words_j
                diff = words_i ^ words_j
                
                # If they share most words but differ in one
                if len(shared) > 0 and len(diff) <= 2:
                    vis_i, lang_i = self.get_features(img_i, text_i)
                    vis_j, lang_j = self.get_features(img_j, text_j)
                    
                    # Correct match should be higher than cross match
                    vis_i_act = self.atl.forward(vis_i)
                    vis_j_act = self.atl.forward(vis_j)
                    lang_i_act = self.atl.forward(lang_i)
                    lang_j_act = self.atl.forward(lang_j)
                    
                    # Matched
                    sim_ii = F.cosine_similarity(vis_i_act.unsqueeze(0), lang_i_act.unsqueeze(0)).item()
                    sim_jj = F.cosine_similarity(vis_j_act.unsqueeze(0), lang_j_act.unsqueeze(0)).item()
                    
                    # Swapped
                    sim_ij = F.cosine_similarity(vis_i_act.unsqueeze(0), lang_j_act.unsqueeze(0)).item()
                    sim_ji = F.cosine_similarity(vis_j_act.unsqueeze(0), lang_i_act.unsqueeze(0)).item()
                    
                    # Score: correct matches should be higher
                    score = (sim_ii > sim_ij) + (sim_ii > sim_ji) + (sim_jj > sim_ij) + (sim_jj > sim_ji)
                    swap_scores.append(score / 4)
        
        if len(swap_scores) == 0:
            return -1.0  # Not applicable
        
        return np.mean(swap_scores)
    
    def random_baseline_test(
        self,
        test_samples: List[Tuple[torch.Tensor, str]],
        n_trials: int = 5,
    ) -> Dict[str, float]:
        """
        Compare learned model to random baseline.
        
        FROM CHPL LESSON: Always compare to random to catch artifacts.
        """
        # Learned model
        learned_metrics = self.compute_discrimination(test_samples)
        learned_disc = learned_metrics['discrimination']
        
        # Random baselines
        random_discs = []
        for _ in range(n_trials):
            # Create random ATL
            random_atl = ATLSemanticHub(
                n_prototypes=self.atl.n_prototypes,
                feature_dim=self.atl.feature_dim,
            ).to(self.device)
            
            # Compute discrimination with random ATL
            random_matched = []
            random_mismatched = []
            
            features = []
            for image, text in test_samples:
                vis_code, lang_emb = self.get_features(image, text)
                vis_act = random_atl.forward(vis_code)
                lang_act = random_atl.forward(lang_emb)
                features.append((vis_act, lang_act))
            
            for vis_act, lang_act in features:
                sim = F.cosine_similarity(vis_act.unsqueeze(0), lang_act.unsqueeze(0)).item()
                random_matched.append(sim)
            
            for i, (vis_act, _) in enumerate(features):
                for j, (_, lang_act) in enumerate(features):
                    if i != j:
                        sim = F.cosine_similarity(vis_act.unsqueeze(0), lang_act.unsqueeze(0)).item()
                        random_mismatched.append(sim)
            
            random_disc = np.mean(random_matched) - np.mean(random_mismatched)
            random_discs.append(random_disc)
        
        random_disc = np.mean(random_discs)
        improvement = learned_disc - random_disc
        
        return {
            'learned_discrimination': learned_disc,
            'random_discrimination': random_disc,
            'improvement_over_random': improvement,
        }
    
    def validate(
        self,
        test_samples: List[Tuple[torch.Tensor, str]],
        include_random_baseline: bool = True,
    ) -> Dict[str, float]:
        """
        Full validation suite.
        
        Args:
            test_samples: List of (image, text) matched pairs
            include_random_baseline: Run random baseline comparison
        
        Returns:
            Dictionary of all metrics
        """
        results = {}
        
        # Core discrimination
        disc_metrics = self.compute_discrimination(test_samples)
        results.update(disc_metrics)
        
        # Swap generalization
        swap_score = self.test_swap_generalization(test_samples)
        if swap_score >= 0:
            results['swap_generalization'] = swap_score
        
        # Random baseline
        if include_random_baseline:
            baseline_metrics = self.random_baseline_test(test_samples)
            results.update(baseline_metrics)
        
        # Capacity
        results['effective_capacity'] = self.atl.get_effective_capacity()
        
        return results
    
    def check_pass(self, metrics: Dict[str, float]) -> Tuple[str, str]:
        """
        Determine pass/fail status.
        
        Returns:
            (status, detailed_message)
        """
        disc = metrics.get('discrimination', 0)
        
        if disc > self.EXCELLENT_THRESHOLD:
            status = 'EXCELLENT'
            symbol = '✓✓'
        elif disc > self.GOOD_THRESHOLD:
            status = 'GOOD'
            symbol = '✓'
        elif disc > self.MARGINAL_THRESHOLD:
            status = 'MARGINAL'
            symbol = '⚠'
        else:
            status = 'FAIL'
            symbol = '✗'
        
        # Build detailed message
        lines = [
            f"Discrimination: {disc:.4f} {symbol}",
            f"  Matched mean: {metrics.get('mean_matched', 0):.4f} (±{metrics.get('std_matched', 0):.4f})",
            f"  Mismatched mean: {metrics.get('mean_mismatched', 0):.4f} (±{metrics.get('std_mismatched', 0):.4f})",
        ]
        
        if 'improvement_over_random' in metrics:
            imp = metrics['improvement_over_random']
            imp_symbol = '✓' if imp > 0.05 else ('⚠' if imp > 0 else '✗')
            lines.append(f"Improvement over random: {imp:.4f} {imp_symbol}")
        
        if 'swap_generalization' in metrics:
            swap = metrics['swap_generalization']
            swap_symbol = '✓' if swap > 0.7 else ('⚠' if swap > 0.5 else '✗')
            lines.append(f"Swap generalization: {swap:.4f} {swap_symbol}")
        
        lines.append(f"Effective capacity: {metrics.get('effective_capacity', 0):.2%}")
        
        message = '\n'.join(lines)
        
        return status, message
    
    def visualize(
        self,
        metrics: Dict[str, float],
        save_dir: str,
    ):
        """Generate validation visualizations."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Discrimination plot
        if 'matched_sims' in metrics and 'mismatched_sims' in metrics:
            plot_discrimination_results(
                metrics['matched_sims'],
                metrics['mismatched_sims'],
                title='ATL Binding Discrimination',
                save_path=str(save_dir / 'atl_discrimination.png'),
                show=False,
            )
        
        # Capacity
        plot_capacity_usage(
            self.atl.usage_count,
            title='ATL Capacity Usage',
            save_path=str(save_dir / 'atl_capacity.png'),
            show=False,
        )


def validate_atl(
    atl_path: str,
    v1_path: str,
    test_data: List[Tuple[torch.Tensor, str]],
    device: str = 'cpu',
    save_dir: Optional[str] = None,
    a1_path: Optional[str] = None,
) -> Tuple[str, Dict[str, float]]:
    """
    Validate ATL model from checkpoint.
    
    THE CRITICAL TEST.
    
    Args:
        atl_path: Path to ATL checkpoint
        v1_path: Path to V1 checkpoint
        test_data: List of (image, text) pairs
        device: Device
        save_dir: Directory for visualizations
        a1_path: Optional A1 checkpoint path
    
    Returns:
        (status, metrics) where status is EXCELLENT/GOOD/MARGINAL/FAIL
    """
    # Load models
    v1 = V1SparseCoding()
    v1_state = torch.load(v1_path, map_location=device)
    v1.load_state_dict(v1_state['model_state_dict'])
    
    atl = ATLSemanticHub()
    atl_state = torch.load(atl_path, map_location=device)
    atl.load_state_dict(atl_state['model_state_dict'])
    
    language = LanguageEncoder(load_pretrained=False)
    
    a1 = None
    if a1_path:
        a1 = A1SparseCoding()
        a1_state = torch.load(a1_path, map_location=device)
        a1.load_state_dict(a1_state['model_state_dict'])
    
    # Validate
    validator = ATLValidator(atl, v1, language, a1, device)
    metrics = validator.validate(test_data)
    status, message = validator.check_pass(metrics)
    
    print("=" * 60)
    print("ATL BINDING VALIDATION - THE CRITICAL TEST")
    print("=" * 60)
    print(message)
    print("=" * 60)
    print(f"FINAL STATUS: {status}")
    print("=" * 60)
    
    if status == 'EXCELLENT':
        print("🎉 EXCELLENT! Ready for publication and trimodal extension.")
    elif status == 'GOOD':
        print("✓ GOOD! Bio-plausible target achieved. Continue to trimodal.")
    elif status == 'MARGINAL':
        print("⚠ MARGINAL. Analyze and consider improvements before proceeding.")
    else:
        print("✗ FAIL. Document negative result. Analyze binding mechanism.")
    
    if save_dir:
        validator.visualize(metrics, save_dir)
    
    return status, metrics
