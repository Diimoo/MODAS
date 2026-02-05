"""
ATL Semantic Hub V2 - Redesigned Cross-Modal Binding

Key architectural changes from V1:
1. SPARSE TOP-K activation (not sigmoid - avoids collapse)
2. MODALITY PROJECTION HEADS (maps different spaces to shared space)
3. PROPER CONTRASTIVE LOSS (InfoNCE-style, learns alignment)
4. BINDING IN PROJECTED SPACE (not prototype space)

The fundamental insight: ATL must CREATE alignment between modalities,
not just preserve pre-existing correlation. This requires:
- Learning to project different modalities into a shared space
- Contrastive loss that pulls matched pairs together, pushes mismatched apart
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from collections import deque


class ATLSemanticHubV2(nn.Module):
    """
    Redesigned ATL semantic hub for cross-modal binding.
    
    Key differences from V1:
    - Uses learnable projection heads per modality
    - Sparse top-k activation instead of sigmoid
    - InfoNCE-style contrastive learning
    - Binding computed in projected shared space
    
    Args:
        n_prototypes: Number of semantic prototypes (default: 200)
        feature_dim: Input feature dimension from V1/A1/Lang (default: 128)
        shared_dim: Dimension of shared semantic space (default: 64)
        topk: Number of active prototypes (sparsity control, default: 10)
        temperature: Contrastive loss temperature (default: 0.1)
        lr_base: Base learning rate (default: 0.01)
        memory_size: Temporal memory buffer size (default: 100)
    """
    
    def __init__(
        self,
        n_prototypes: int = 200,
        feature_dim: int = 128,
        shared_dim: int = 64,
        topk: int = 10,
        temperature: float = 0.1,
        lr_base: float = 0.01,
        memory_size: int = 100,
    ):
        super().__init__()
        
        self.n_prototypes = n_prototypes
        self.feature_dim = feature_dim
        self.shared_dim = shared_dim
        self.topk = topk
        self.temperature = temperature
        self.lr_base = lr_base
        
        # MODALITY-SPECIFIC PROJECTION HEADS
        # These learn to map each modality into a shared semantic space
        self.proj_visual = nn.Linear(feature_dim, shared_dim, bias=False)
        self.proj_audio = nn.Linear(feature_dim, shared_dim, bias=False)
        self.proj_language = nn.Linear(feature_dim, shared_dim, bias=False)
        
        # Initialize projections with small weights
        for proj in [self.proj_visual, self.proj_audio, self.proj_language]:
            nn.init.orthogonal_(proj.weight, gain=0.5)
        
        # Semantic prototypes in SHARED space
        self.register_buffer(
            'prototypes',
            self._init_prototypes()
        )
        
        # Usage tracking for meta-plasticity
        self.register_buffer(
            'usage_count',
            torch.zeros(n_prototypes)
        )
        
        # Temporal memory buffers (store PROJECTED features)
        self.memory_vis: deque = deque(maxlen=memory_size)
        self.memory_aud: deque = deque(maxlen=memory_size)
        self.memory_lang: deque = deque(maxlen=memory_size)
        
        # Training statistics
        self.training_step = 0
        self.loss_history: List[float] = []
        self.margin_history: List[float] = []
    
    def _init_prototypes(self) -> torch.Tensor:
        """Initialize prototypes with random unit-norm vectors in shared space."""
        prototypes = torch.randn(self.n_prototypes, self.shared_dim)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        return prototypes
    
    def project(self, features: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Project features from modality-specific space to shared semantic space.
        
        Args:
            features: Input features (feature_dim,) or (batch, feature_dim)
            modality: 'visual', 'audio', or 'language'
        
        Returns:
            Projected features (shared_dim,) or (batch, shared_dim), normalized
        """
        squeeze = features.dim() == 1
        if squeeze:
            features = features.unsqueeze(0)
        
        if modality == 'visual':
            projected = self.proj_visual(features)
        elif modality == 'audio':
            projected = self.proj_audio(features)
        elif modality == 'language':
            projected = self.proj_language(features)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Normalize to unit sphere
        projected = F.normalize(projected, p=2, dim=1)
        
        if squeeze:
            projected = projected.squeeze(0)
        
        return projected
    
    def get_sparse_activation(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get sparse top-k activation over prototypes.
        
        Unlike sigmoid (which clusters around 0.5), top-k is truly sparse
        and provides discriminative representations.
        
        Args:
            features: Projected features in shared space (shared_dim,) or (batch, shared_dim)
        
        Returns:
            Sparse activation (n_prototypes,) or (batch, n_prototypes)
        """
        squeeze = features.dim() == 1
        if squeeze:
            features = features.unsqueeze(0)
        
        # Cosine similarity to prototypes
        similarities = features @ self.prototypes.T  # (batch, n_prototypes)
        
        # Top-k sparse activation
        topk_vals, topk_idx = similarities.topk(self.topk, dim=1)
        
        # Create sparse activation vector
        activation = torch.zeros_like(similarities)
        activation.scatter_(1, topk_idx, F.softmax(topk_vals / self.temperature, dim=1))
        
        if squeeze:
            activation = activation.squeeze(0)
        
        return activation
    
    def get_semantic_embedding(self, features: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Get semantic embedding for features.
        
        Projects to shared space and computes prototype-weighted embedding.
        
        Args:
            features: Raw features (feature_dim,)
            modality: 'visual', 'audio', or 'language'
        
        Returns:
            Semantic embedding (shared_dim,)
        """
        projected = self.project(features, modality)
        activation = self.get_sparse_activation(projected)
        
        # Weighted sum of prototypes
        embedding = activation @ self.prototypes  # (shared_dim,)
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding
    
    def compute_cross_modal_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        modality1: str = 'visual',
        modality2: str = 'language',
    ) -> torch.Tensor:
        """
        Compute cross-modal similarity in PROJECTED shared space.
        
        This is the key metric for discrimination.
        """
        proj1 = self.project(features1, modality1)
        proj2 = self.project(features2, modality2)
        
        return F.cosine_similarity(proj1.unsqueeze(0), proj2.unsqueeze(0)).squeeze()
    
    def bind(
        self,
        vis_features: torch.Tensor,
        lang_features: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal binding via contrastive learning.
        
        Projects both modalities to shared space and applies InfoNCE-style loss.
        
        Args:
            vis_features: Visual features (feature_dim,)
            lang_features: Language features (feature_dim,)
            audio_features: Optional audio features (feature_dim,)
        
        Returns:
            Tuple of (similarity, contrastive_loss)
        """
        # Project to shared space
        vis_proj = self.project(vis_features, 'visual')
        lang_proj = self.project(lang_features, 'language')
        
        # Positive similarity (matched pair in projected space)
        pos_sim = F.cosine_similarity(
            vis_proj.unsqueeze(0),
            lang_proj.unsqueeze(0)
        ).squeeze()
        
        # Gather negatives from memory
        neg_lang_projs = []
        if len(self.memory_lang) > 0:
            n_neg = min(20, len(self.memory_lang))
            indices = np.random.choice(len(self.memory_lang), n_neg, replace=False)
            for idx in indices:
                neg_lang_projs.append(self.memory_lang[idx])
        
        # InfoNCE-style contrastive loss
        if len(neg_lang_projs) > 0:
            neg_lang_stack = torch.stack(neg_lang_projs)  # (n_neg, shared_dim)
            neg_sims = (vis_proj.unsqueeze(0) @ neg_lang_stack.T).squeeze()  # (n_neg,)
            
            # InfoNCE: -log(exp(pos/t) / (exp(pos/t) + sum(exp(neg/t))))
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sims]) / self.temperature
            labels = torch.zeros(1, dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits.unsqueeze(0), labels)
            
            margin = pos_sim - neg_sims.max()
        else:
            loss = -pos_sim  # Early training: just maximize positive similarity
            margin = pos_sim
        
        # Update projection heads via gradient (requires grad context)
        if self.training and vis_features.requires_grad:
            loss.backward(retain_graph=True)
        
        # Store projected features in memory
        self.memory_vis.append(vis_proj.detach().clone())
        self.memory_lang.append(lang_proj.detach().clone())
        
        if audio_features is not None:
            aud_proj = self.project(audio_features, 'audio')
            self.memory_aud.append(aud_proj.detach().clone())
        
        # Update prototypes via Hebbian learning
        self._update_prototypes(vis_proj.detach(), lang_proj.detach(), margin.detach())
        
        # Track statistics
        self.training_step += 1
        self.loss_history.append(loss.item())
        self.margin_history.append(margin.item())
        
        return pos_sim, margin
    
    def _update_prototypes(
        self,
        vis_proj: torch.Tensor,
        lang_proj: torch.Tensor,
        margin: torch.Tensor,
    ):
        """
        Update prototypes via Hebbian learning.
        
        Pull prototypes toward projected features for positive pairs.
        """
        # Combined feature (mean of projections)
        combined = F.normalize((vis_proj + lang_proj) / 2, p=2, dim=0)
        
        # Find closest prototype
        similarities = combined @ self.prototypes.T
        closest_idx = similarities.argmax()
        
        # Hebbian update: pull prototype toward combined feature
        modulator = torch.sigmoid(margin)  # 0-1 based on margin quality
        lr = self.lr_base * modulator
        
        delta = lr * (combined - self.prototypes[closest_idx])
        self.prototypes[closest_idx] = F.normalize(
            self.prototypes[closest_idx] + delta, p=2, dim=0
        )
        
        # Update usage
        self.usage_count[closest_idx] += 1
    
    def forward(
        self,
        features: torch.Tensor,
        modality: str = 'visual'
    ) -> torch.Tensor:
        """
        Compute sparse semantic activation for features.
        
        Projects to shared space and returns sparse top-k activation.
        """
        projected = self.project(features, modality)
        return self.get_sparse_activation(projected)
    
    def get_effective_capacity(self, threshold: float = 0.01) -> float:
        """Compute fraction of prototypes that have been used."""
        return (self.usage_count > threshold).float().mean().item()


def create_atl_v2(feature_dim: int = 128, **kwargs) -> ATLSemanticHubV2:
    """Factory function to create ATL V2 with proper initialization."""
    return ATLSemanticHubV2(feature_dim=feature_dim, **kwargs)
