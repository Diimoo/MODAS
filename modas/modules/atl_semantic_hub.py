"""
ATL Semantic Hub - Cross-Modal Binding via Contrastive Hebbian Learning

Binds V1, A1, and Language features into unified semantic space.
Implements Contrastive Hebbian Learning (CHL) with improvements from CHPL.

Key features (FROM LESSONS LEARNED):
- NO softmax (creates artifacts) - use sigmoid instead
- Temporal memory for hard negatives
- Contrastive margin for discrimination
- Three-factor Hebbian (pre × post × modulator)
- Meta-plasticity to prevent prototype collapse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque


class ATLSemanticHub(nn.Module):
    """
    Anterior Temporal Lobe semantic hub for cross-modal binding.
    
    Uses Contrastive Hebbian Learning to bind visual, auditory, and
    language representations into a unified semantic space.
    
    Args:
        n_prototypes: Number of semantic prototypes (default: 200)
        feature_dim: Feature dimension from modules (default: 128)
        temperature: Activation temperature (default: 0.2)
        lr_base: Base learning rate (default: 0.01)
        meta_beta: Meta-plasticity factor (default: 0.999)
        memory_size: Temporal memory buffer size (default: 100)
        n_negatives: Number of negatives for contrastive (default: 20)
    """
    
    def __init__(
        self,
        n_prototypes: int = 200,
        feature_dim: int = 128,
        temperature: float = 0.2,
        lr_base: float = 0.01,
        meta_beta: float = 0.999,
        memory_size: int = 100,
        n_negatives: int = 20,
    ):
        super().__init__()
        
        self.n_prototypes = n_prototypes
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.lr_base = lr_base
        self.meta_beta = meta_beta
        self.n_negatives = n_negatives
        
        # Semantic prototypes (learnable)
        self.register_buffer(
            'prototypes',
            self._init_prototypes()
        )
        
        # Usage tracking for meta-plasticity (FROM MBM)
        self.register_buffer(
            'usage_count',
            torch.zeros(n_prototypes)
        )
        
        # Temporal memory buffers (FROM CHPL Phase 0.5)
        self.memory_vis: deque = deque(maxlen=memory_size)
        self.memory_aud: deque = deque(maxlen=memory_size)
        self.memory_lang: deque = deque(maxlen=memory_size)
        
        # Training statistics
        self.training_step = 0
        self.similarity_history: List[float] = []
        self.margin_history: List[float] = []
    
    def _init_prototypes(self) -> torch.Tensor:
        """Initialize prototypes with random unit-norm vectors."""
        prototypes = torch.randn(self.n_prototypes, self.feature_dim)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        return prototypes
    
    def forward(
        self,
        features: torch.Tensor,
        modality: str = 'visual'
    ) -> torch.Tensor:
        """
        Compute semantic activation for input features.
        
        Uses temperature-scaled sigmoid (NOT softmax - FROM CHPL LESSON).
        
        Args:
            features: Input features (feature_dim,) or (batch, feature_dim)
            modality: Modality type ('visual', 'audio', 'language')
        
        Returns:
            Semantic activation (n_prototypes,) or (batch, n_prototypes)
        """
        # Handle single vs batch
        if features.dim() == 1:
            features = features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Ensure features are normalized
        features = F.normalize(features, p=2, dim=1)
        
        # Similarity to prototypes
        similarities = features @ self.prototypes.T  # (batch, n_prototypes)
        
        # Temperature-scaled sigmoid activation (NOT softmax!)
        # FROM CHPL LESSON: Softmax creates 0.7 artifact
        activations = torch.sigmoid(similarities / self.temperature)
        
        if squeeze_output:
            activations = activations.squeeze(0)
        
        return activations
    
    def get_semantic_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get semantic embedding by weighting prototypes by activation.
        
        Args:
            features: Input features (feature_dim,)
        
        Returns:
            Semantic embedding (feature_dim,)
        """
        activations = self.forward(features)  # (n_prototypes,)
        
        # Weight prototypes by activation
        weighted = activations.unsqueeze(1) * self.prototypes  # (n_prototypes, feature_dim)
        embedding = weighted.sum(dim=0)  # (feature_dim,)
        
        return F.normalize(embedding, p=2, dim=0)
    
    def bind(
        self,
        vis_features: torch.Tensor,
        lang_features: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal binding via Contrastive Hebbian Learning.
        
        Args:
            vis_features: Visual features (feature_dim,)
            lang_features: Language features (feature_dim,)
            audio_features: Optional audio features (feature_dim,)
        
        Returns:
            Tuple of (positive_similarity, contrastive_margin)
        """
        # Ensure features are normalized
        vis_features = F.normalize(vis_features, p=2, dim=0)
        lang_features = F.normalize(lang_features, p=2, dim=0)
        if audio_features is not None:
            audio_features = F.normalize(audio_features, p=2, dim=0)
        
        # Get activations
        vis_act = self.forward(vis_features, 'visual')
        lang_act = self.forward(lang_features, 'language')
        
        # Positive similarity (matched pair)
        pos_sim = F.cosine_similarity(
            vis_act.unsqueeze(0),
            lang_act.unsqueeze(0)
        ).squeeze()
        
        # Negative similarities from temporal memory (FROM CHPL)
        neg_sims = self._get_negative_similarities(vis_act, 'language')
        
        # Contrastive margin (FROM CHPL)
        if len(neg_sims) > 0:
            max_neg = max(neg_sims)
            margin = pos_sim - max_neg
        else:
            margin = pos_sim  # Early training, no negatives yet
        
        # Three-factor Hebbian update
        modulator = torch.tanh(margin)  # Bounded [-1, 1]
        
        self._hebbian_update(vis_features, vis_act, modulator)
        self._hebbian_update(lang_features, lang_act, modulator)
        
        if audio_features is not None:
            aud_act = self.forward(audio_features, 'audio')
            self._hebbian_update(audio_features, aud_act, modulator)
        
        # Update memory buffers
        self.memory_vis.append(vis_features.detach().clone())
        self.memory_lang.append(lang_features.detach().clone())
        if audio_features is not None:
            self.memory_aud.append(audio_features.detach().clone())
        
        # Track statistics
        self.training_step += 1
        self.similarity_history.append(pos_sim.item())
        self.margin_history.append(margin.item())
        
        return pos_sim, margin
    
    def _get_negative_similarities(
        self,
        query_act: torch.Tensor,
        modality: str
    ) -> List[float]:
        """
        Get negative similarities from memory buffer.
        
        Args:
            query_act: Query activation (n_prototypes,)
            modality: Modality to sample negatives from
        
        Returns:
            List of negative similarities
        """
        if modality == 'language':
            memory = self.memory_lang
        elif modality == 'visual':
            memory = self.memory_vis
        elif modality == 'audio':
            memory = self.memory_aud
        else:
            return []
        
        if len(memory) == 0:
            return []
        
        # Sample negatives
        n_samples = min(self.n_negatives, len(memory))
        indices = np.random.choice(len(memory), n_samples, replace=False)
        
        neg_sims = []
        for idx in indices:
            neg_features = memory[idx]
            neg_act = self.forward(neg_features, modality)
            sim = F.cosine_similarity(
                query_act.unsqueeze(0),
                neg_act.unsqueeze(0)
            ).item()
            neg_sims.append(sim)
        
        return neg_sims
    
    def _hebbian_update(
        self,
        features: torch.Tensor,
        activations: torch.Tensor,
        modulator: torch.Tensor,
    ):
        """
        Three-factor Hebbian update for prototypes.
        
        update = eta * modulator * activation * (features - prototype)
        
        Args:
            features: Input features (feature_dim,)
            activations: Prototype activations (n_prototypes,)
            modulator: Learning modulator (scalar)
        """
        for i in range(self.n_prototypes):
            # Hebbian correlation: activation × feature
            hebbian = activations[i] * features
            
            # Direction: move toward features
            delta = hebbian - activations[i] * self.prototypes[i]
            
            # Meta-plasticity (FROM MBM)
            effective_lr = self.lr_base / (1.0 + self.meta_beta * self.usage_count[i])
            
            # Modulated update (three-factor Hebbian)
            self.prototypes[i] = self.prototypes[i] + effective_lr * modulator * delta
        
        # Re-normalize prototypes
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        
        # Update usage
        self.usage_count = self.usage_count + activations.abs()
    
    def bind_batch(
        self,
        vis_batch: torch.Tensor,
        lang_batch: torch.Tensor,
        audio_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch cross-modal binding.
        
        Args:
            vis_batch: Visual features (batch, feature_dim)
            lang_batch: Language features (batch, feature_dim)
            audio_batch: Optional audio features (batch, feature_dim)
        
        Returns:
            Tuple of (positive_similarities, margins) both (batch,)
        """
        batch_size = vis_batch.shape[0]
        
        pos_sims = []
        margins = []
        
        for i in range(batch_size):
            vis = vis_batch[i]
            lang = lang_batch[i]
            audio = audio_batch[i] if audio_batch is not None else None
            
            pos_sim, margin = self.bind(vis, lang, audio)
            pos_sims.append(pos_sim)
            margins.append(margin)
        
        return torch.stack(pos_sims), torch.stack(margins)
    
    def compute_cross_modal_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-modal similarity through ATL space.
        
        Args:
            features1: First modality features (feature_dim,)
            features2: Second modality features (feature_dim,)
        
        Returns:
            Similarity score (scalar)
        """
        act1 = self.forward(features1)
        act2 = self.forward(features2)
        
        return F.cosine_similarity(act1.unsqueeze(0), act2.unsqueeze(0)).squeeze()
    
    def retrieve_by_modality(
        self,
        query_features: torch.Tensor,
        target_memory: str = 'visual',
        topk: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Cross-modal retrieval: find similar items in target modality.
        
        Args:
            query_features: Query features (feature_dim,)
            target_memory: Target modality memory ('visual', 'audio', 'language')
            topk: Number of results to return
        
        Returns:
            List of (memory_index, similarity) tuples
        """
        if target_memory == 'visual':
            memory = self.memory_vis
        elif target_memory == 'audio':
            memory = self.memory_aud
        elif target_memory == 'language':
            memory = self.memory_lang
        else:
            return []
        
        if len(memory) == 0:
            return []
        
        query_act = self.forward(query_features)
        
        similarities = []
        for idx, mem_features in enumerate(memory):
            mem_act = self.forward(mem_features)
            sim = F.cosine_similarity(
                query_act.unsqueeze(0),
                mem_act.unsqueeze(0)
            ).item()
            similarities.append((idx, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: -x[1])
        
        return similarities[:topk]
    
    def get_discrimination(
        self,
        matched_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        mismatched_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> float:
        """
        Compute discrimination score.
        
        discrimination = mean(matched_sim) - mean(mismatched_sim)
        
        Args:
            matched_pairs: List of (vis, lang) matched pairs
            mismatched_pairs: List of (vis, lang) mismatched pairs
        
        Returns:
            Discrimination score
        """
        matched_sims = []
        for vis, lang in matched_pairs:
            sim = self.compute_cross_modal_similarity(vis, lang)
            matched_sims.append(sim.item())
        
        mismatched_sims = []
        for vis, lang in mismatched_pairs:
            sim = self.compute_cross_modal_similarity(vis, lang)
            mismatched_sims.append(sim.item())
        
        disc = np.mean(matched_sims) - np.mean(mismatched_sims)
        return disc
    
    def get_sparsity(self, features: torch.Tensor, threshold: float = 0.1) -> float:
        """Compute activation sparsity."""
        activations = self.forward(features)
        return (activations < threshold).float().mean().item()
    
    def get_effective_capacity(self, threshold: float = 0.01) -> float:
        """Compute effective capacity (fraction of used prototypes)."""
        return (self.usage_count > threshold).float().mean().item()
    
    def reset_memory(self):
        """Clear memory buffers."""
        self.memory_vis.clear()
        self.memory_aud.clear()
        self.memory_lang.clear()
    
    def reset_usage(self):
        """Reset usage counts."""
        self.usage_count.zero_()
    
    def reset_statistics(self):
        """Reset training statistics."""
        self.training_step = 0
        self.similarity_history.clear()
        self.margin_history.clear()
    
    def get_training_stats(self) -> Dict:
        """Get training statistics summary."""
        if len(self.similarity_history) == 0:
            return {'steps': 0}
        
        return {
            'steps': self.training_step,
            'mean_similarity': np.mean(self.similarity_history[-100:]),
            'mean_margin': np.mean(self.margin_history[-100:]),
            'effective_capacity': self.get_effective_capacity(),
        }
