"""
ATL Semantic Hub V4 - Hebbian with Collapse Prevention

Fixes over V3:
1. Anti-Hebbian decorrelation to prevent projection collapse
2. Hard threshold for teaching signals (no noise propagation)
3. Designed for proper generalization testing

The collapse problem in V3: both projections chase each other's output,
converging to a constant function. Fix: add within-modality repulsion.

Update rule:
    Δw = η_attract × (y_other ⊗ x) - η_repel × (ȳ_self ⊗ x) - decay

Where ȳ_self is running mean of self-projections. This ensures:
- Cross-modal attraction: matched pairs → similar outputs
- Within-modal repulsion: different inputs → different outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque


class HebbianProjectionV4(nn.Module):
    """
    Hebbian projection with anti-collapse mechanisms.
    
    Key additions over V3:
    1. Running mean of outputs for decorrelation
    2. Anti-Hebbian repulsion from mean
    3. Hard threshold for updates (no noise propagation)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lr_attract: float = 0.01,
        lr_repel: float = 0.005,
        momentum: float = 0.99,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr_attract = lr_attract
        self.lr_repel = lr_repel
        self.momentum = momentum
        
        # Projection weights (buffer, not parameter)
        self.register_buffer(
            'weight',
            torch.randn(output_dim, input_dim) * 0.1
        )
        
        # Running mean of outputs (for decorrelation)
        self.register_buffer('running_mean', torch.zeros(output_dim))
        self.register_buffer('n_updates', torch.tensor(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and normalize."""
        projected = F.linear(x, self.weight)
        return F.normalize(projected, p=2, dim=-1)
    
    def hebbian_update(
        self,
        input_features: torch.Tensor,
        teaching_signal: torch.Tensor,
        self_output: torch.Tensor,
        margin: float,
        margin_threshold: float = 0.0,
    ):
        """
        Update weights with cross-modal attraction AND within-modal repulsion.
        
        Args:
            input_features: This modality's input (input_dim,)
            teaching_signal: Other modality's output (output_dim,)
            self_output: This modality's current output (output_dim,)
            margin: Contrastive margin for gating
            margin_threshold: Only update if margin > threshold
        """
        # Hard threshold: don't propagate noise early in training
        if margin < margin_threshold:
            return
        
        with torch.no_grad():
            x = input_features.flatten()
            y_other = teaching_signal.flatten()
            y_self = self_output.flatten()
            
            # Update running mean of self-outputs
            self.n_updates += 1
            self.running_mean = (
                self.momentum * self.running_mean + 
                (1 - self.momentum) * y_self
            )
            
            # === CROSS-MODAL ATTRACTION ===
            # Pull projection toward producing outputs like teaching_signal
            attract = torch.outer(y_other, x)
            
            # === WITHIN-MODAL REPULSION ===
            # Push projection away from producing the average output
            # This prevents collapse to a constant function
            repel = torch.outer(self.running_mean, x)
            
            # === OJA-STYLE DECAY ===
            # Prevent weight blow-up: Δw -= y² × w
            decay = (y_other.unsqueeze(1) ** 2) * self.weight
            
            # Combined update
            delta = (
                self.lr_attract * attract
                - self.lr_repel * repel
                - self.lr_attract * decay
            )
            
            self.weight.add_(delta)
            self.weight.clamp_(-2.0, 2.0)


class ATLSemanticHubV4(nn.Module):
    """
    ATL with collapse prevention and proper generalization testing.
    
    Key improvements:
    1. Anti-Hebbian decorrelation prevents projection collapse
    2. Hard margin threshold prevents noise propagation
    3. Exposes train/test split capability for generalization
    """
    
    def __init__(
        self,
        n_prototypes: int = 200,
        feature_dim: int = 128,
        shared_dim: int = 64,
        temperature: float = 0.2,
        lr_attract: float = 0.01,
        lr_repel: float = 0.005,
        lr_proto: float = 0.05,
        margin_threshold: float = 0.0,
        memory_size: int = 100,
    ):
        super().__init__()
        
        self.n_prototypes = n_prototypes
        self.feature_dim = feature_dim
        self.shared_dim = shared_dim
        self.temperature = temperature
        self.lr_proto = lr_proto
        self.margin_threshold = margin_threshold
        
        # Hebbian projections with anti-collapse
        self.proj_visual = HebbianProjectionV4(
            feature_dim, shared_dim, lr_attract, lr_repel
        )
        self.proj_audio = HebbianProjectionV4(
            feature_dim, shared_dim, lr_attract, lr_repel
        )
        self.proj_language = HebbianProjectionV4(
            feature_dim, shared_dim, lr_attract, lr_repel
        )
        
        # Prototypes
        self.register_buffer(
            'prototypes',
            F.normalize(torch.randn(n_prototypes, shared_dim), dim=1)
        )
        self.register_buffer('usage_count', torch.zeros(n_prototypes))
        self.register_buffer('prototype_lr', torch.ones(n_prototypes) * lr_proto)
        
        # Memory for contrastive negatives
        self.memory_vis: deque = deque(maxlen=memory_size)
        self.memory_lang: deque = deque(maxlen=memory_size)
        
        # Statistics
        self.training_step = 0
        self.similarity_history: List[float] = []
        self.margin_history: List[float] = []
        self.collapse_metric_history: List[float] = []
    
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
    
    def get_activation(self, projected: torch.Tensor) -> torch.Tensor:
        """Softmax activation over prototypes."""
        squeeze = projected.dim() == 1
        if squeeze:
            projected = projected.unsqueeze(0)
        
        similarities = projected @ self.prototypes.T
        activations = F.softmax(similarities / self.temperature, dim=1)
        
        if squeeze:
            activations = activations.squeeze(0)
        return activations
    
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
    
    def compute_collapse_metric(self) -> float:
        """
        Measure projection collapse.
        
        Computes variance of projections across recent memory.
        Low variance = collapse to constant function.
        """
        if len(self.memory_vis) < 10:
            return 1.0  # Not enough data
        
        vis_stack = torch.stack(list(self.memory_vis)[-50:])
        lang_stack = torch.stack(list(self.memory_lang)[-50:])
        
        # Variance across samples (should be high if diverse)
        vis_var = vis_stack.var(dim=0).mean().item()
        lang_var = lang_stack.var(dim=0).mean().item()
        
        return (vis_var + lang_var) / 2
    
    def _get_contrastive_margin(
        self,
        vis_proj: torch.Tensor,
        lang_proj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute contrastive margin from memory."""
        pos_sim = F.cosine_similarity(
            vis_proj.unsqueeze(0), lang_proj.unsqueeze(0)
        ).squeeze()
        
        if len(self.memory_lang) > 5:
            n_neg = min(20, len(self.memory_lang))
            indices = np.random.choice(len(self.memory_lang), n_neg, replace=False)
            neg_projs = torch.stack([self.memory_lang[i] for i in indices])
            neg_sims = (vis_proj.unsqueeze(0) @ neg_projs.T).squeeze()
            max_neg = neg_sims.max()
            margin = pos_sim - max_neg
        else:
            margin = pos_sim - 0.5  # Conservative early margin
        
        return pos_sim, margin
    
    def bind(
        self,
        vis_features: torch.Tensor,
        lang_features: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal binding with collapse prevention.
        """
        vis_features = F.normalize(vis_features, p=2, dim=-1)
        lang_features = F.normalize(lang_features, p=2, dim=-1)
        
        # Project
        vis_proj = self.project(vis_features, 'visual')
        lang_proj = self.project(lang_features, 'language')
        
        # Contrastive margin
        pos_sim, margin = self._get_contrastive_margin(vis_proj, lang_proj)
        
        # === HEBBIAN PROJECTION UPDATES WITH ANTI-COLLAPSE ===
        self.proj_visual.hebbian_update(
            input_features=vis_features,
            teaching_signal=lang_proj.detach(),
            self_output=vis_proj.detach(),
            margin=margin.item(),
            margin_threshold=self.margin_threshold,
        )
        
        self.proj_language.hebbian_update(
            input_features=lang_features,
            teaching_signal=vis_proj.detach(),
            self_output=lang_proj.detach(),
            margin=margin.item(),
            margin_threshold=self.margin_threshold,
        )
        
        # Re-project after update
        vis_proj_new = self.project(vis_features, 'visual')
        lang_proj_new = self.project(lang_features, 'language')
        
        # Prototype update (only on positive margin)
        if margin.item() > self.margin_threshold:
            avg_act = (self.get_activation(vis_proj_new) + self.get_activation(lang_proj_new)) / 2
            avg_proj = F.normalize((vis_proj_new + lang_proj_new) / 2, dim=-1)
            
            with torch.no_grad():
                for i in range(self.n_prototypes):
                    act = avg_act[i].item()
                    if act > 0.01:
                        lr = self.prototype_lr[i].item()
                        delta = lr * act * (avg_proj - self.prototypes[i])
                        self.prototypes[i] = F.normalize(
                            self.prototypes[i] + delta, dim=-1
                        )
                        self.usage_count[i] += act
                        self.prototype_lr[i] *= 0.9999
        
        # Store in memory
        self.memory_vis.append(vis_proj_new.detach().clone())
        self.memory_lang.append(lang_proj_new.detach().clone())
        
        # Track statistics
        self.training_step += 1
        self.similarity_history.append(pos_sim.item())
        self.margin_history.append(margin.item())
        
        if self.training_step % 100 == 0:
            self.collapse_metric_history.append(self.compute_collapse_metric())
        
        return pos_sim, margin
    
    def forward(self, features: torch.Tensor, modality: str = 'visual') -> torch.Tensor:
        """Semantic activation."""
        projected = self.project(features, modality)
        return self.get_activation(projected)
    
    def get_effective_capacity(self, threshold: float = 0.01) -> float:
        """Fraction of used prototypes."""
        return (self.usage_count > threshold).float().mean().item()
    
    def get_stats(self) -> Dict:
        """Training statistics."""
        return {
            'step': self.training_step,
            'mean_sim': np.mean(self.similarity_history[-100:]) if self.similarity_history else 0,
            'mean_margin': np.mean(self.margin_history[-100:]) if self.margin_history else 0,
            'collapse_metric': self.collapse_metric_history[-1] if self.collapse_metric_history else 1.0,
            'effective_capacity': self.get_effective_capacity(),
        }


def create_atl_v4(feature_dim: int = 128, **kwargs) -> ATLSemanticHubV4:
    """Factory function."""
    return ATLSemanticHubV4(feature_dim=feature_dim, **kwargs)
