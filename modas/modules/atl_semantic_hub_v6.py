"""
ATL Semantic Hub V6 - Heteroassociative Hebbian Binding

Architecture insight: Don't project to shared space. Associate.

The brain doesn't align visual and language representations by projecting
them to a common space. It binds them through temporal co-occurrence in
the anterior temporal lobe, using associative Hebbian learning.

Architecture:
    Visual features → competitive visual prototypes (P_v)
    Language features → competitive language prototypes (P_l)
    Binding matrix W (P_v × P_l): Hebbian co-occurrence

    Update: W += η × vis_act ⊗ lang_act (co-active → bind)
    Similarity: vis_act @ W @ lang_act (through association)

Why this works:
- No projection learning (the hard problem that V3-V5 couldn't solve)
- Pure Hebbian (outer product of activations)
- Scales well: W is n_vis_proto × n_lang_proto, not feature_dim × feature_dim
- Naturally handles different-dimension modalities
- Biologically faithful: competitive layers + associative binding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from collections import deque


class CompetitiveLayer(nn.Module):
    """
    Competitive Hebbian layer that clusters inputs into prototypes.
    
    Uses winner-take-most activation (temperature-scaled softmax)
    and Hebbian prototype learning with meta-plasticity.
    """
    
    def __init__(
        self,
        feature_dim: int,
        n_prototypes: int,
        temperature: float = 0.2,
        lr: float = 0.05,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_prototypes = n_prototypes
        self.temperature = temperature
        self.lr = lr
        
        # Prototypes (buffers, not parameters)
        self.register_buffer(
            'prototypes',
            F.normalize(torch.randn(n_prototypes, feature_dim), dim=1)
        )
        self.register_buffer('usage_count', torch.zeros(n_prototypes))
        self.register_buffer('proto_lr', torch.ones(n_prototypes) * lr)
    
    def forward(self, x: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        Sparse top-k activation over prototypes.
        
        Only the top-k most similar prototypes get non-zero activation.
        This is critical: dense activations make the binding matrix uniform.
        
        Args:
            x: (*, feature_dim) normalized input features
            k: number of active prototypes per input
        Returns:
            (*, n_prototypes) sparse activation pattern
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.normalize(x, p=2, dim=-1)
        sims = x @ self.prototypes.T  # (B, n_prototypes)
        
        # Top-k: zero out everything except top k
        topk_vals, topk_idx = sims.topk(k, dim=-1)  # (B, k)
        
        # Sparse activation: softmax only over top-k, rest = 0
        sparse_act = torch.zeros_like(sims)
        topk_softmax = F.softmax(topk_vals / self.temperature, dim=-1)
        sparse_act.scatter_(1, topk_idx, topk_softmax)
        
        return sparse_act
    
    def learn_batch(self, x: torch.Tensor, k: int = 5):
        """
        Competitive Hebbian learning: move prototypes toward inputs.
        
        Args:
            x: (B, feature_dim) normalized input features
            k: number of active prototypes per input
        """
        with torch.no_grad():
            x = F.normalize(x, p=2, dim=-1)
            acts = self.forward(x, k=k)  # (B, n_prototypes) — sparse
            
            # Weighted average of inputs per prototype
            # weighted_input[p] = sum_b acts[b, p] * x[b] / sum_b acts[b, p]
            weight_sum = acts.sum(dim=0)  # (n_prototypes,)
            weighted_input = acts.T @ x   # (n_prototypes, feature_dim)
            
            active = weight_sum > 0.01
            if active.any():
                target = weighted_input[active] / weight_sum[active].unsqueeze(1)
                lr = self.proto_lr[active].unsqueeze(1)
                
                # Move prototypes toward their assigned inputs
                self.prototypes[active] = F.normalize(
                    self.prototypes[active] + lr * (target - self.prototypes[active]),
                    dim=-1
                )
                self.usage_count[active] += weight_sum[active]
                
                # Meta-plasticity: reduce LR for frequently used prototypes
                self.proto_lr[active] *= 0.9999
            
            # Resurrect dead prototypes
            dead = self.usage_count < 0.1
            if dead.any() and self.usage_count.sum() > 10:
                n_dead = dead.sum().item()
                # Replace dead prototypes with random perturbations of active ones
                alive_idx = (~dead).nonzero(as_tuple=True)[0]
                if len(alive_idx) > 0:
                    donors = alive_idx[torch.randint(len(alive_idx), (n_dead,))]
                    noise = torch.randn_like(self.prototypes[dead]) * 0.1
                    self.prototypes[dead] = F.normalize(
                        self.prototypes[donors] + noise, dim=-1
                    )
                    self.proto_lr[dead] = self.lr
                    self.usage_count[dead] = 0.1


class ATLSemanticHubV6(nn.Module):
    """
    Heteroassociative Hebbian binding between competitive layers.
    
    No projections. No shared space. Pure associative binding.
    
    Components:
    1. Visual competitive layer: clusters visual features
    2. Language competitive layer: clusters language features
    3. Audio competitive layer: clusters audio features
    4. Binding matrices: Hebbian association between layer pairs
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        n_vis_protos: int = 100,
        n_lang_protos: int = 100,
        n_aud_protos: int = 100,
        temperature: float = 0.2,
        lr_proto: float = 0.05,
        lr_bind: float = 0.1,
        lr_unbind: float = 0.02,
        memory_size: int = 50,
        k: int = 5,
    ):
        super().__init__()
        
        self.temperature = temperature
        self.lr_bind = lr_bind
        self.lr_unbind = lr_unbind
        self.k = k
        
        # Competitive layers (one per modality)
        self.vis_layer = CompetitiveLayer(
            feature_dim, n_vis_protos, temperature, lr_proto
        )
        self.lang_layer = CompetitiveLayer(
            feature_dim, n_lang_protos, temperature, lr_proto
        )
        self.aud_layer = CompetitiveLayer(
            feature_dim, n_aud_protos, temperature, lr_proto
        )
        
        # Binding matrices (Hebbian association)
        # W_vl[i,j] = strength of association between vis_proto i and lang_proto j
        self.register_buffer(
            'W_vl', torch.zeros(n_vis_protos, n_lang_protos)
        )
        self.register_buffer(
            'W_va', torch.zeros(n_vis_protos, n_aud_protos)
        )
        self.register_buffer(
            'W_la', torch.zeros(n_lang_protos, n_aud_protos)
        )
        
        # Memory for contrastive anti-Hebbian
        self.memory_vis: deque = deque(maxlen=memory_size)
        self.memory_lang: deque = deque(maxlen=memory_size)
        
        self.training_step = 0
    
    def get_activation(self, features: torch.Tensor, modality: str) -> torch.Tensor:
        """Get sparse competitive activation for a modality."""
        if modality == 'visual':
            return self.vis_layer(features, k=self.k)
        elif modality == 'language':
            return self.lang_layer(features, k=self.k)
        elif modality == 'audio':
            return self.aud_layer(features, k=self.k)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def compute_cross_modal_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        modality1: str = 'visual',
        modality2: str = 'language',
    ) -> torch.Tensor:
        """
        Similarity through associative binding.
        
        sim = act1 @ W @ act2
        """
        act1 = self.get_activation(features1, modality1)
        act2 = self.get_activation(features2, modality2)
        
        if act1.dim() == 2:
            act1 = act1.squeeze(0)
        if act2.dim() == 2:
            act2 = act2.squeeze(0)
        
        W = self._get_binding_matrix(modality1, modality2)
        return act1 @ W @ act2
    
    def _get_binding_matrix(self, mod1: str, mod2: str) -> torch.Tensor:
        """Get the binding matrix between two modalities."""
        pair = tuple(sorted([mod1, mod2]))
        if pair == ('language', 'visual'):
            return self.W_vl if mod1 == 'visual' else self.W_vl.T
        elif pair == ('audio', 'visual'):
            return self.W_va if mod1 == 'visual' else self.W_va.T
        elif pair == ('audio', 'language'):
            return self.W_la if mod1 == 'language' else self.W_la.T
        else:
            raise ValueError(f"Unknown modality pair: {mod1}, {mod2}")
    
    def bind_batch(
        self,
        vis_features: torch.Tensor,
        lang_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched cross-modal binding.
        
        1. Learn competitive prototypes (cluster inputs)
        2. Compute activations
        3. Hebbian bind: strengthen co-occurring prototype pairs
        4. Anti-Hebbian unbind: weaken non-co-occurring pairs (from memory)
        """
        B = vis_features.shape[0]
        vis_features = F.normalize(vis_features, p=2, dim=-1)
        lang_features = F.normalize(lang_features, p=2, dim=-1)
        
        # Step 1: Learn competitive prototypes
        self.vis_layer.learn_batch(vis_features, k=self.k)
        self.lang_layer.learn_batch(lang_features, k=self.k)
        
        # Step 2: Get sparse activations (only top-k prototypes active)
        vis_act = self.vis_layer(vis_features, k=self.k)    # (B, n_vis_protos) sparse
        lang_act = self.lang_layer(lang_features, k=self.k)  # (B, n_lang_protos) sparse
        
        with torch.no_grad():
            # Step 3: Hebbian bind (positive pairs)
            # Average outer product over batch
            bind_update = (vis_act.T @ lang_act) / B  # (n_vis, n_lang)
            self.W_vl += self.lr_bind * bind_update
            
            # Step 4: Anti-Hebbian unbind (negative pairs from memory)
            if len(self.memory_lang) >= 5:
                n_neg = min(20, len(self.memory_lang))
                neg_indices = np.random.choice(len(self.memory_lang), n_neg, replace=False)
                neg_lang_acts = torch.stack([self.memory_lang[i] for i in neg_indices])
                # Unbind: weaken association between current vis and random lang
                # unbind[i,j] = mean_b(vis_act[b,i]) * mean_k(neg_lang[k,j])
                unbind_update = torch.outer(
                    vis_act.mean(dim=0), neg_lang_acts.mean(dim=0)
                )
                self.W_vl -= self.lr_unbind * unbind_update
            
            # Clamp and normalize binding matrix
            self.W_vl.clamp_(0.0, None)  # Non-negative associations
            # Normalize rows to prevent unbounded growth
            row_norms = self.W_vl.sum(dim=1, keepdim=True).clamp(min=1e-8)
            self.W_vl /= row_norms.clamp(min=1.0)
        
        # Store activations in memory for future anti-Hebbian
        for i in range(min(B, 10)):
            self.memory_vis.append(vis_act[i].detach().clone())
            self.memory_lang.append(lang_act[i].detach().clone())
        
        # Compute similarity for monitoring
        pos_sims = (vis_act * (lang_act @ self.W_vl.T)).sum(dim=1)  # (B,)
        
        # Contrastive margin
        if len(self.memory_lang) >= 5:
            n_neg = min(20, len(self.memory_lang))
            neg_idx = np.random.choice(len(self.memory_lang), n_neg, replace=False)
            neg_la = torch.stack([self.memory_lang[i] for i in neg_idx])
            neg_sims = (vis_act @ self.W_vl @ neg_la.T)  # (B, n_neg)
            max_neg, _ = neg_sims.max(dim=1)
            margins = pos_sims - max_neg
        else:
            margins = pos_sims
        
        self.training_step += B
        return pos_sims.mean(), margins.mean()
    
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
        """Get activation pattern for a modality."""
        return self.get_activation(features, modality)
    
    def get_effective_capacity(self, threshold: float = 0.01) -> float:
        """Fraction of used visual prototypes."""
        return (self.vis_layer.usage_count > threshold).float().mean().item()
    
    def get_stats(self) -> Dict:
        """Training statistics."""
        return {
            'step': self.training_step,
            'vis_capacity': (self.vis_layer.usage_count > 0.01).float().mean().item(),
            'lang_capacity': (self.lang_layer.usage_count > 0.01).float().mean().item(),
            'bind_sparsity': (self.W_vl > 0.01).float().mean().item(),
            'bind_max': self.W_vl.max().item(),
        }
