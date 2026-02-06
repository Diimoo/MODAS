"""
ATL Semantic Hub V5 - Bottleneck + Covariance Decorrelation

Fixes V4's failures:
1. V4's running-mean anti-Hebbian degenerates in high dims (mean → 0)
2. V4's Hebbian cross-correlation has low SNR in high dims (too many params)

V5 architecture:
    input (d_in) → bottleneck (d_bn) → shared (d_shared)

Key changes:
1. Bottleneck reduces parameters so Hebbian rule has enough SNR
2. Running covariance decorrelation (second-order, not first-order)
3. Covariance term: Δw -= η_decor × (C_self - I) × W
   This pushes output covariance toward identity = decorrelation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque


class HebbianBottleneckProjection(nn.Module):
    """
    Two-stage Hebbian projection with bottleneck and covariance decorrelation.
    
    Stage 1: input (d_in) → bottleneck (d_bn) via Hebbian
    Stage 2: bottleneck (d_bn) → shared (d_shared) via Hebbian
    
    Both stages use covariance decorrelation instead of running mean.
    """
    
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        output_dim: int,
        lr: float = 0.01,
        lr_decor: float = 0.005,
        cov_momentum: float = 0.99,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.lr = lr
        self.lr_decor = lr_decor
        self.cov_momentum = cov_momentum
        
        # Stage 1: input → bottleneck
        self.register_buffer(
            'W1', torch.randn(bottleneck_dim, input_dim) * (2.0 / input_dim) ** 0.5
        )
        # Stage 2: bottleneck → output
        self.register_buffer(
            'W2', torch.randn(output_dim, bottleneck_dim) * (2.0 / bottleneck_dim) ** 0.5
        )
        
        # Running covariance of bottleneck outputs (for stage 1 decorrelation)
        self.register_buffer('cov_bn', torch.eye(bottleneck_dim))
        # Running covariance of final outputs (for stage 2 decorrelation)
        self.register_buffer('cov_out', torch.eye(output_dim))
        
        self.register_buffer('n_updates', torch.tensor(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project through bottleneck and normalize."""
        h = F.linear(x, self.W1)            # → (*, bottleneck_dim)
        h = F.relu(h)                        # Sparse bottleneck
        out = F.linear(h, self.W2)           # → (*, output_dim)
        return F.normalize(out, p=2, dim=-1)
    
    def forward_stages(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both bottleneck and output activations."""
        h = F.linear(x, self.W1)
        h = F.relu(h)
        out = F.linear(h, self.W2)
        return h, F.normalize(out, p=2, dim=-1)
    
    def hebbian_update_batch(
        self,
        input_features: torch.Tensor,
        teaching_signal: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Batched Hebbian update with covariance decorrelation.
        
        Args:
            input_features: (B, input_dim)
            teaching_signal: (B, output_dim) from other modality
            mask: (B,) bool — which samples to use
        """
        if mask.sum() == 0:
            return
        
        with torch.no_grad():
            x = input_features[mask]           # (N, input_dim)
            y_teach = teaching_signal[mask]     # (N, output_dim)
            N = x.shape[0]
            
            # Forward through stages
            h = F.relu(F.linear(x, self.W1))   # (N, bottleneck_dim)
            y_self = F.normalize(F.linear(h, self.W2), dim=-1)  # (N, output_dim)
            
            # === UPDATE RUNNING COVARIANCES ===
            self.n_updates += N
            
            # Bottleneck covariance
            h_centered = h - h.mean(dim=0, keepdim=True)
            cov_bn_batch = (h_centered.T @ h_centered) / max(N - 1, 1)
            self.cov_bn = (
                self.cov_momentum * self.cov_bn +
                (1 - self.cov_momentum) * cov_bn_batch
            )
            
            # Output covariance
            y_centered = y_self - y_self.mean(dim=0, keepdim=True)
            cov_out_batch = (y_centered.T @ y_centered) / max(N - 1, 1)
            self.cov_out = (
                self.cov_momentum * self.cov_out +
                (1 - self.cov_momentum) * cov_out_batch
            )
            
            # === STAGE 2: W2 update ===
            # Attraction: align output with teaching signal
            attract_2 = (y_teach.T @ h) / N  # (output_dim, bottleneck_dim)
            
            # Decorrelation: push output covariance toward identity
            # (C_out - I) @ W2 penalizes correlated outputs
            decor_2 = (self.cov_out - torch.eye(
                self.output_dim, device=x.device
            )) @ self.W2  # (output_dim, bottleneck_dim)
            
            # Oja-style decay
            decay_2 = ((y_teach ** 2).mean(dim=0).unsqueeze(1)) * self.W2
            
            self.W2 += self.lr * attract_2 - self.lr_decor * decor_2 - self.lr * 0.01 * decay_2
            
            # === STAGE 1: W1 update ===
            # Backproject teaching signal to bottleneck target
            h_target = F.relu(y_teach @ self.W2)  # (N, bottleneck_dim)
            
            # Attraction: align bottleneck with target
            attract_1 = (h_target.T @ x) / N  # (bottleneck_dim, input_dim)
            
            # Decorrelation: push bottleneck covariance toward identity
            decor_1 = (self.cov_bn - torch.eye(
                self.bottleneck_dim, device=x.device
            )) @ self.W1  # (bottleneck_dim, input_dim)
            
            # Oja-style decay
            decay_1 = ((h_target ** 2).mean(dim=0).unsqueeze(1)) * self.W1
            
            self.W1 += self.lr * attract_1 - self.lr_decor * decor_1 - self.lr * 0.01 * decay_1
            
            # Clamp weights
            self.W1.clamp_(-3.0, 3.0)
            self.W2.clamp_(-3.0, 3.0)


class ATLSemanticHubV5(nn.Module):
    """
    ATL V5: Bottleneck projections + covariance decorrelation.
    
    Addresses V4 failures:
    - Scale: bottleneck reduces parameter count for high-dim inputs
    - Sensitivity: covariance decorrelation is more principled than running mean
    - Generalization: proper second-order statistics
    """
    
    def __init__(
        self,
        n_prototypes: int = 200,
        feature_dim: int = 128,
        bottleneck_dim: int = 16,
        shared_dim: int = 64,
        temperature: float = 0.2,
        lr: float = 0.01,
        lr_decor: float = 0.005,
        lr_proto: float = 0.05,
        margin_threshold: float = -0.3,
        memory_size: int = 100,
    ):
        super().__init__()
        
        self.n_prototypes = n_prototypes
        self.feature_dim = feature_dim
        self.bottleneck_dim = bottleneck_dim
        self.shared_dim = shared_dim
        self.temperature = temperature
        self.lr_proto = lr_proto
        self.margin_threshold = margin_threshold
        
        # Bottleneck projections
        self.proj_visual = HebbianBottleneckProjection(
            feature_dim, bottleneck_dim, shared_dim, lr, lr_decor
        )
        self.proj_audio = HebbianBottleneckProjection(
            feature_dim, bottleneck_dim, shared_dim, lr, lr_decor
        )
        self.proj_language = HebbianBottleneckProjection(
            feature_dim, bottleneck_dim, shared_dim, lr, lr_decor
        )
        
        # Prototypes in shared space
        self.register_buffer(
            'prototypes',
            F.normalize(torch.randn(n_prototypes, shared_dim), dim=1)
        )
        self.register_buffer('usage_count', torch.zeros(n_prototypes))
        self.register_buffer('prototype_lr', torch.ones(n_prototypes) * lr_proto)
        
        # Statistics
        self.training_step = 0
    
    def project(self, features: torch.Tensor, modality: str) -> torch.Tensor:
        """Project to shared space via bottleneck."""
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
        Fully vectorized cross-modal binding.
        """
        B = vis_features.shape[0]
        vis_features = F.normalize(vis_features, p=2, dim=-1)
        lang_features = F.normalize(lang_features, p=2, dim=-1)
        
        # Project
        vis_proj = self.proj_visual(vis_features)
        lang_proj = self.proj_language(lang_features)
        
        # Positive similarity and contrastive margin
        pos_sims = F.cosine_similarity(vis_proj, lang_proj, dim=1)
        sim_matrix = vis_proj @ lang_proj.T
        mask_diag = ~torch.eye(B, dtype=torch.bool, device=vis_features.device)
        neg_sims = sim_matrix.masked_fill(~mask_diag, -1e9)
        max_neg, _ = neg_sims.max(dim=1)
        margins = pos_sims - max_neg
        
        # Threshold mask
        update_mask = margins > self.margin_threshold
        
        # Hebbian updates with covariance decorrelation
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
        """Single-sample bind (wraps bind_batch)."""
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
        """Fraction of used prototypes."""
        return (self.usage_count > threshold).float().mean().item()
    
    def get_stats(self) -> Dict:
        """Training statistics."""
        return {
            'step': self.training_step,
            'effective_capacity': self.get_effective_capacity(),
        }
