"""
ATL Semantic Hub V3 - Biologically Plausible Cross-Modal Binding

Key architectural principles (staying true to Hebbian):
1. NO BACKPROP - all learning is local
2. Hebbian Cross-Correlation Learning for projections
3. Competitive Hebbian Learning for prototypes
4. Three-factor rule with contrastive modulation

The insight: To align two unrelated spaces (V1 sparse codes, Word2Vec),
we need to learn the PROJECTIONS, not just prototypes. But we can do this
with Hebbian rules by using cross-modal correlation:

    Δw_vis = η × (lang_proj ⊗ vis_input) - decorrelation term
    
This is biologically plausible: the "teaching signal" for visual projection
comes from the language pathway, and vice versa.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque


class HebbianProjection(nn.Module):
    """
    A projection layer that learns via Hebbian cross-correlation.
    
    Instead of backprop, uses:
    - Cross-modal teaching signal (other modality's projection)
    - Oja-style weight decay to prevent blow-up
    - Lateral decorrelation to learn independent features
    """
    
    def __init__(self, input_dim: int, output_dim: int, lr: float = 0.01):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        
        # Initialize weights (small random, will learn)
        # NOT using nn.Linear because we'll update weights manually
        self.register_buffer(
            'weight',
            torch.randn(output_dim, input_dim) * 0.1
        )
        
        # Running statistics for decorrelation
        self.register_buffer('output_cov', torch.eye(output_dim) * 0.1)
        self.cov_momentum = 0.99
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to output space."""
        projected = F.linear(x, self.weight)
        return F.normalize(projected, p=2, dim=-1)
    
    def hebbian_update(
        self,
        input_features: torch.Tensor,
        teaching_signal: torch.Tensor,
        modulator: float = 1.0,
    ):
        """
        Update weights via Hebbian cross-correlation learning.
        
        Args:
            input_features: This modality's raw features (input_dim,)
            teaching_signal: Other modality's projected output (output_dim,)
            modulator: Contrastive modulation signal (from margin)
        
        The update rule:
            Δw = η × modulator × (teaching_signal ⊗ input) - weight_decay × w
        
        This is biologically plausible:
        - teaching_signal = "what the other modality thinks this should be"
        - input = "what this modality sees"
        - modulator = "was this a good match?" (from contrastive margin)
        """
        with torch.no_grad():
            # Ensure 1D
            if input_features.dim() > 1:
                input_features = input_features.flatten()
            if teaching_signal.dim() > 1:
                teaching_signal = teaching_signal.flatten()
            
            # Cross-correlation update: teaching × input^T
            # This pulls the projection toward producing outputs similar to teaching_signal
            outer = torch.outer(teaching_signal, input_features)
            
            # Oja-style weight decay to prevent blow-up
            # Δw = η(y×x - y²×w) where y = teaching_signal
            y_sq = teaching_signal.unsqueeze(1) ** 2
            weight_decay = y_sq * self.weight
            
            # Combined update with modulator
            delta = self.lr * modulator * (outer - weight_decay)
            self.weight.add_(delta)
            
            # Keep weights bounded
            self.weight.clamp_(-2.0, 2.0)


class ATLSemanticHubV3(nn.Module):
    """
    Biologically plausible ATL using only Hebbian learning.
    
    Architecture:
    1. Hebbian projection heads (learn via cross-correlation)
    2. Competitive prototype layer (winner-take-most)
    3. Three-factor Hebbian update (pre × post × modulator)
    
    No backprop anywhere. All updates are local.
    """
    
    def __init__(
        self,
        n_prototypes: int = 200,
        feature_dim: int = 128,
        shared_dim: int = 64,
        temperature: float = 0.2,
        lr_proj: float = 0.01,
        lr_proto: float = 0.05,
        memory_size: int = 100,
    ):
        super().__init__()
        
        self.n_prototypes = n_prototypes
        self.feature_dim = feature_dim
        self.shared_dim = shared_dim
        self.temperature = temperature
        self.lr_proto = lr_proto
        
        # HEBBIAN PROJECTION HEADS (learn via cross-correlation)
        self.proj_visual = HebbianProjection(feature_dim, shared_dim, lr=lr_proj)
        self.proj_audio = HebbianProjection(feature_dim, shared_dim, lr=lr_proj)
        self.proj_language = HebbianProjection(feature_dim, shared_dim, lr=lr_proj)
        
        # Semantic prototypes in shared space
        self.register_buffer(
            'prototypes',
            F.normalize(torch.randn(n_prototypes, shared_dim), dim=1)
        )
        
        # Usage tracking for meta-plasticity
        self.register_buffer('usage_count', torch.zeros(n_prototypes))
        self.register_buffer('prototype_lr', torch.ones(n_prototypes) * lr_proto)
        
        # Temporal memory for contrastive negatives
        self.memory_vis: deque = deque(maxlen=memory_size)
        self.memory_lang: deque = deque(maxlen=memory_size)
        
        # Statistics
        self.training_step = 0
        self.similarity_history: List[float] = []
        self.margin_history: List[float] = []
    
    def project(self, features: torch.Tensor, modality: str) -> torch.Tensor:
        """Project features to shared space via Hebbian projection."""
        if modality == 'visual':
            return self.proj_visual(features)
        elif modality == 'audio':
            return self.proj_audio(features)
        elif modality == 'language':
            return self.proj_language(features)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def get_activation(self, projected: torch.Tensor) -> torch.Tensor:
        """
        Compute activation over prototypes.
        
        Uses softmax with temperature (competitive/lateral inhibition).
        """
        squeeze = projected.dim() == 1
        if squeeze:
            projected = projected.unsqueeze(0)
        
        # Cosine similarity to prototypes
        similarities = projected @ self.prototypes.T
        
        # Softmax competition (lateral inhibition)
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
        """Compute similarity in projected shared space."""
        proj1 = self.project(features1, modality1)
        proj2 = self.project(features2, modality2)
        return F.cosine_similarity(proj1.unsqueeze(0), proj2.unsqueeze(0)).squeeze()
    
    def _get_contrastive_margin(
        self,
        vis_proj: torch.Tensor,
        lang_proj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute contrastive margin from temporal memory.
        
        Returns (positive_similarity, margin).
        """
        pos_sim = F.cosine_similarity(
            vis_proj.unsqueeze(0),
            lang_proj.unsqueeze(0)
        ).squeeze()
        
        # Get negative similarities from memory
        if len(self.memory_lang) > 5:
            n_neg = min(20, len(self.memory_lang))
            indices = np.random.choice(len(self.memory_lang), n_neg, replace=False)
            neg_projs = torch.stack([self.memory_lang[i] for i in indices])
            neg_sims = (vis_proj.unsqueeze(0) @ neg_projs.T).squeeze()
            max_neg = neg_sims.max()
            margin = pos_sim - max_neg
        else:
            margin = pos_sim
        
        return pos_sim, margin
    
    def bind(
        self,
        vis_features: torch.Tensor,
        lang_features: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal binding via Hebbian learning.
        
        ALL LEARNING IS LOCAL:
        1. Project both modalities to shared space
        2. Compute contrastive margin (modulation signal)
        3. Update projections via cross-correlation Hebbian rule
        4. Update prototypes via three-factor Hebbian rule
        """
        # Normalize inputs
        vis_features = F.normalize(vis_features, p=2, dim=-1)
        lang_features = F.normalize(lang_features, p=2, dim=-1)
        
        # Project to shared space
        vis_proj = self.project(vis_features, 'visual')
        lang_proj = self.project(lang_features, 'language')
        
        # Compute contrastive margin (teaching signal quality)
        pos_sim, margin = self._get_contrastive_margin(vis_proj, lang_proj)
        
        # Convert margin to modulation signal: good margin -> positive update
        # Use sigmoid to bound between 0 and 1
        modulator = torch.sigmoid(margin * 5).item()  # Scale margin for sigmoid
        
        # ===== HEBBIAN PROJECTION UPDATE =====
        # Key insight: Each projection learns from the OTHER modality's output
        # Visual projection learns to produce outputs similar to language projection
        # Language projection learns to produce outputs similar to visual projection
        
        self.proj_visual.hebbian_update(
            input_features=vis_features,
            teaching_signal=lang_proj.detach(),  # Teach visual to match language
            modulator=modulator,
        )
        
        self.proj_language.hebbian_update(
            input_features=lang_features,
            teaching_signal=vis_proj.detach(),  # Teach language to match visual
            modulator=modulator,
        )
        
        # ===== HEBBIAN PROTOTYPE UPDATE =====
        # Three-factor rule: pre × post × modulator
        
        # Re-project after projection update
        vis_proj_new = self.project(vis_features, 'visual')
        lang_proj_new = self.project(lang_features, 'language')
        
        # Get activations (post-synaptic)
        vis_act = self.get_activation(vis_proj_new)
        lang_act = self.get_activation(lang_proj_new)
        
        # Combined activation and projection
        avg_act = (vis_act + lang_act) / 2
        avg_proj = F.normalize((vis_proj_new + lang_proj_new) / 2, dim=-1)
        
        # Update prototypes (Hebbian with meta-plasticity)
        with torch.no_grad():
            for i in range(self.n_prototypes):
                act = avg_act[i].item()
                
                if act > 0.01:  # Only update active prototypes
                    # Meta-plasticity: reduce LR for frequently used prototypes
                    lr = self.prototype_lr[i].item() * modulator
                    
                    # Hebbian: move prototype toward input
                    delta = lr * act * (avg_proj - self.prototypes[i])
                    self.prototypes[i] = F.normalize(
                        self.prototypes[i] + delta, dim=-1
                    )
                    
                    # Update usage and decay LR
                    self.usage_count[i] += act
                    self.prototype_lr[i] *= 0.9999
        
        # Store in memory for future contrastive signals
        self.memory_vis.append(vis_proj_new.detach().clone())
        self.memory_lang.append(lang_proj_new.detach().clone())
        
        # Track statistics
        self.training_step += 1
        self.similarity_history.append(pos_sim.item())
        self.margin_history.append(margin.item())
        
        return pos_sim, margin
    
    def forward(
        self,
        features: torch.Tensor,
        modality: str = 'visual'
    ) -> torch.Tensor:
        """Compute semantic activation for features."""
        projected = self.project(features, modality)
        return self.get_activation(projected)
    
    def get_effective_capacity(self, threshold: float = 0.01) -> float:
        """Compute fraction of prototypes that have been used."""
        return (self.usage_count > threshold).float().mean().item()
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'step': self.training_step,
            'mean_sim': np.mean(self.similarity_history[-100:]) if self.similarity_history else 0,
            'mean_margin': np.mean(self.margin_history[-100:]) if self.margin_history else 0,
            'effective_capacity': self.get_effective_capacity(),
            'active_prototypes': (self.usage_count > 0.1).sum().item(),
        }


def create_atl_v3(feature_dim: int = 128, **kwargs) -> ATLSemanticHubV3:
    """Factory function to create ATL V3."""
    return ATLSemanticHubV3(feature_dim=feature_dim, **kwargs)
