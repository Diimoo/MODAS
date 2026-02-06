"""
ATL Semantic Hub V7 - Centered Cross-Covariance Hebbian Learning

The key insight: center inputs/outputs before Hebbian update.
This is equivalent to learning cross-covariance (Hebbian CCA),
which naturally prevents collapse without any anti-Hebbian term.

Why centering works:
- In high dims, mean of normalized vectors → 0
- V4's running mean repulsion degenerates to a bias term
- Centering removes the mean, so the update only captures
  co-variation between modalities, not shared bias

Update rule (per batch):
    x_c = x - mean(x)           # centered input
    y_c = y_other - mean(y_other) # centered teaching signal
    ΔW = η × (y_c.T @ x_c) / N  # cross-covariance

This is:
- Biologically plausible (centering = subtracting mean firing rate)
- No running statistics (batch-only computation)
- No hyperparameters to tune (no η_repel, no momentum)
- Scale-invariant (works the same at 64-dim and 256-dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class CenteredHebbianProjection(nn.Module):
    """
    Hebbian projection with centered cross-covariance learning.
    
    Simpler than V4's HebbianProjectionV4:
    - No running mean
    - No anti-Hebbian term
    - No momentum parameter
    - Just centering + cross-covariance
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lr: float = 0.01,
        weight_decay: float = 0.001,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.register_buffer(
            'weight',
            torch.randn(output_dim, input_dim) * (2.0 / input_dim) ** 0.5
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and normalize."""
        projected = F.linear(x, self.weight)
        return F.normalize(projected, p=2, dim=-1)
    
    def hebbian_update_batch(
        self,
        input_features: torch.Tensor,
        teaching_signal: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Centered cross-covariance Hebbian update.
        
        Args:
            input_features: (B, input_dim)
            teaching_signal: (B, output_dim) from other modality
            mask: (B,) bool — which samples pass threshold
        """
        if mask.sum() < 2:  # Need ≥2 for meaningful centering
            return
        
        with torch.no_grad():
            x = input_features[mask]       # (N, input_dim)
            y = teaching_signal[mask]       # (N, output_dim)
            N = x.shape[0]
            
            # Center (subtract batch mean)
            x_c = x - x.mean(dim=0, keepdim=True)
            y_c = y - y.mean(dim=0, keepdim=True)
            
            # Cross-covariance update
            cov_update = (y_c.T @ x_c) / N  # (output_dim, input_dim)
            
            # Weight decay (Oja-style)
            decay = self.weight_decay * self.weight
            
            # Update
            self.weight += self.lr * cov_update - decay
            
            # Clamp for stability
            self.weight.clamp_(-3.0, 3.0)


class ATLSemanticHubV7(nn.Module):
    """
    ATL V7: Centered cross-covariance Hebbian projections.
    
    Compared to V4:
    - Simpler (fewer hyperparams: no η_repel, no momentum)
    - Scale-invariant (centering works at any dimension)
    - No collapse (centering removes the degenerate mode)
    """
    
    def __init__(
        self,
        n_prototypes: int = 200,
        feature_dim: int = 128,
        shared_dim: int = 64,
        temperature: float = 0.2,
        lr: float = 0.01,
        weight_decay: float = 0.001,
        lr_proto: float = 0.05,
        margin_threshold: float = -0.3,
    ):
        super().__init__()
        
        self.n_prototypes = n_prototypes
        self.feature_dim = feature_dim
        self.shared_dim = shared_dim
        self.temperature = temperature
        self.lr_proto = lr_proto
        self.margin_threshold = margin_threshold
        
        # Centered Hebbian projections
        self.proj_visual = CenteredHebbianProjection(
            feature_dim, shared_dim, lr, weight_decay
        )
        self.proj_audio = CenteredHebbianProjection(
            feature_dim, shared_dim, lr, weight_decay
        )
        self.proj_language = CenteredHebbianProjection(
            feature_dim, shared_dim, lr, weight_decay
        )
        
        # Prototypes in shared space
        self.register_buffer(
            'prototypes',
            F.normalize(torch.randn(n_prototypes, shared_dim), dim=1)
        )
        self.register_buffer('usage_count', torch.zeros(n_prototypes))
        self.register_buffer('prototype_lr', torch.ones(n_prototypes) * lr_proto)
        
        self.training_step = 0
    
    def project(self, features: torch.Tensor, modality: str) -> torch.Tensor:
        """Project to shared space."""
        if modality == 'visual':
            return self.proj_visual(features)
        elif modality == 'audio':
            return self.proj_audio(features)
        elif modality == 'language':
            return self.proj_language(features)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def compute_cross_modal_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        modality1: str = 'visual',
        modality2: str = 'language',
    ) -> torch.Tensor:
        """Similarity in projected space."""
        proj1 = self.project(features1, modality1)
        proj2 = self.project(features2, modality2)
        return F.cosine_similarity(proj1.unsqueeze(0), proj2.unsqueeze(0)).squeeze()
    
    def bind_batch(
        self,
        vis_features: torch.Tensor,
        lang_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched cross-modal binding with centered Hebbian learning.
        """
        B = vis_features.shape[0]
        vis_features = F.normalize(vis_features, p=2, dim=-1)
        lang_features = F.normalize(lang_features, p=2, dim=-1)
        
        # Project
        vis_proj = self.proj_visual(vis_features)
        lang_proj = self.proj_language(lang_features)
        
        # In-batch contrastive margin
        pos_sims = F.cosine_similarity(vis_proj, lang_proj, dim=1)
        sim_matrix = vis_proj @ lang_proj.T
        mask_diag = ~torch.eye(B, dtype=torch.bool, device=vis_features.device)
        neg_sims = sim_matrix.masked_fill(~mask_diag, -1e9)
        max_neg, _ = neg_sims.max(dim=1)
        margins = pos_sims - max_neg
        
        # Threshold mask
        update_mask = margins > self.margin_threshold
        
        # Centered cross-covariance Hebbian updates
        self.proj_visual.hebbian_update_batch(
            input_features=vis_features,
            teaching_signal=lang_proj.detach(),
            mask=update_mask,
        )
        self.proj_language.hebbian_update_batch(
            input_features=lang_features,
            teaching_signal=vis_proj.detach(),
            mask=update_mask,
        )
        
        # Prototype update (vectorized)
        if update_mask.any():
            vis_proj_new = self.proj_visual(vis_features)
            lang_proj_new = self.proj_language(lang_features)
            self._update_prototypes_batch(vis_proj_new, lang_proj_new, update_mask)
        
        self.training_step += B
        return pos_sims.mean(), margins.mean()
    
    def _update_prototypes_batch(
        self,
        vis_proj: torch.Tensor,
        lang_proj: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Vectorized prototype update."""
        if mask.sum() == 0:
            return
        
        with torch.no_grad():
            v = vis_proj[mask]
            l = lang_proj[mask]
            avg_proj = F.normalize((v + l) / 2, dim=-1)
            
            vis_act = F.softmax((v @ self.prototypes.T) / self.temperature, dim=1)
            lang_act = F.softmax((l @ self.prototypes.T) / self.temperature, dim=1)
            avg_act = (vis_act + lang_act) / 2
            avg_act = avg_act * (avg_act > 0.01).float()
            
            weight_sum = avg_act.sum(dim=0)
            weighted_proj = avg_act.T @ avg_proj
            
            active = weight_sum > 0.01
            if active.any():
                lr = self.prototype_lr[active]
                w = weight_sum[active]
                target = weighted_proj[active] / w.unsqueeze(1)
                scale = (lr * w).unsqueeze(1)
                delta = scale * (target - self.prototypes[active])
                self.prototypes[active] = F.normalize(
                    self.prototypes[active] + delta, dim=-1
                )
                self.usage_count[active] += w
                self.prototype_lr[active] *= 0.9999
    
    def bind(
        self,
        vis_features: torch.Tensor,
        lang_features: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-sample bind."""
        vis = vis_features.unsqueeze(0) if vis_features.dim() == 1 else vis_features
        lang = lang_features.unsqueeze(0) if lang_features.dim() == 1 else lang_features
        return self.bind_batch(vis, lang)
    
    def forward(self, features: torch.Tensor, modality: str = 'visual') -> torch.Tensor:
        """Semantic activation."""
        projected = self.project(features, modality)
        if projected.dim() == 1:
            projected = projected.unsqueeze(0)
        similarities = projected @ self.prototypes.T
        return F.softmax(similarities / self.temperature, dim=1).squeeze(0)
    
    def get_effective_capacity(self, threshold: float = 0.01) -> float:
        return (self.usage_count > threshold).float().mean().item()
    
    def get_stats(self) -> Dict:
        return {
            'step': self.training_step,
            'effective_capacity': self.get_effective_capacity(),
        }
