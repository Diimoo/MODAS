#!/usr/bin/env python3
"""
ATL V4 Sensitivity and Scale Analysis (GPU-accelerated)

1. Sensitivity: η_repel at 0.25x..4x baseline
2. Scale: up to 256-dim features, 50+ concepts
3. Push test discrimination to ≥ 0.15
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from modas.modules.atl_semantic_hub_v4 import ATLSemanticHubV4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def create_concept_data(
    n_concepts: int,
    instances_per_concept: int,
    feature_dim: int,
    noise_scale: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create train/test tensors on DEVICE.
    Returns: (train_vis, train_lang, train_labels, test_vis, test_lang, test_labels)
    """
    concept_protos = F.normalize(torch.randn(n_concepts, feature_dim, device=DEVICE), dim=1)
    
    train_n = int(instances_per_concept * 0.75)
    test_n = instances_per_concept - train_n
    
    train_vis, train_lang, train_labels = [], [], []
    test_vis, test_lang, test_labels = [], [], []
    
    for c in range(n_concepts):
        all_vis = F.normalize(
            concept_protos[c].unsqueeze(0) + noise_scale * torch.randn(instances_per_concept, feature_dim, device=DEVICE),
            dim=1
        )
        all_lang = F.normalize(
            concept_protos[c].unsqueeze(0) + noise_scale * torch.randn(instances_per_concept, feature_dim, device=DEVICE),
            dim=1
        )
        perm = torch.randperm(instances_per_concept, device=DEVICE)
        all_vis = all_vis[perm]
        all_lang = all_lang[perm]
        
        train_vis.append(all_vis[:train_n])
        train_lang.append(all_lang[:train_n])
        train_labels.append(torch.full((train_n,), c, device=DEVICE))
        test_vis.append(all_vis[train_n:])
        test_lang.append(all_lang[train_n:])
        test_labels.append(torch.full((test_n,), c, device=DEVICE))
    
    return (
        torch.cat(train_vis), torch.cat(train_lang), torch.cat(train_labels),
        torch.cat(test_vis), torch.cat(test_lang), torch.cat(test_labels),
    )


def compute_discrimination_batched(
    atl: ATLSemanticHubV4,
    vis: torch.Tensor,
    lang: torch.Tensor,
    n_eval: int = 30,
) -> float:
    """Batched discrimination computation."""
    n = min(len(vis), n_eval)
    v = vis[:n]
    l = lang[:n]
    
    with torch.no_grad():
        v_proj = atl.proj_visual(F.normalize(v, dim=1))   # (n, shared_dim)
        l_proj = atl.proj_language(F.normalize(l, dim=1))  # (n, shared_dim)
        
        # Matched: diagonal of sim matrix
        sim_matrix = v_proj @ l_proj.T  # (n, n)
        matched = sim_matrix.diag().mean().item()
        
        # Mismatched: off-diagonal
        mask = ~torch.eye(n, dtype=torch.bool, device=vis.device)
        mismatched = sim_matrix[mask].mean().item()
    
    return matched - mismatched


def compute_variance_batched(atl: ATLSemanticHubV4, vis: torch.Tensor) -> float:
    """Batched projection variance."""
    with torch.no_grad():
        projs = atl.proj_visual(F.normalize(vis[:50], dim=1))
        return projs.var(dim=0).mean().item()


def experiment1_sensitivity():
    """Test η_repel sensitivity at 0.25x, 0.5x, 1x, 2x, 3x, 4x."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: η_repel Sensitivity Analysis")
    print("=" * 70)
    
    base_lr_repel = 0.005
    multipliers = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
    
    train_vis, train_lang, train_lbl, test_vis, test_lang, test_lbl = create_concept_data(
        n_concepts=16, instances_per_concept=20, feature_dim=64
    )
    
    print(f"\nConfig: 16 concepts, 64-dim features, 32-dim shared, 3000 epochs")
    print(f"Train: {len(train_vis)}, Test: {len(test_vis)}")
    print(f"\nBase η_repel = {base_lr_repel}")
    print(f"\n{'Mult':<8} {'η_repel':<10} {'Var':<12} {'Train':<12} {'Test':<12} {'Status'}")
    print("-" * 66)
    
    results = []
    for mult in multipliers:
        lr_repel = base_lr_repel * mult
        
        atl = ATLSemanticHubV4(
            n_prototypes=100, feature_dim=64, shared_dim=32,
            lr_attract=0.01, lr_repel=lr_repel, margin_threshold=-0.3,
        ).to(DEVICE)
        
        t0 = time.time()
        for epoch in range(3000):
            perm = torch.randperm(len(train_vis), device=DEVICE)
            atl.bind_batch(train_vis[perm], train_lang[perm])
        elapsed = time.time() - t0
        
        var = compute_variance_batched(atl, train_vis)
        train_disc = compute_discrimination_batched(atl, train_vis, train_lang)
        test_disc = compute_discrimination_batched(atl, test_vis, test_lang)
        
        if var < 0.001:
            status = "COLLAPSED"
        elif test_disc < 0.05:
            status = "WEAK"
        elif test_disc < 0.15:
            status = "PARTIAL"
        else:
            status = "GOOD"
        
        results.append((mult, lr_repel, var, train_disc, test_disc, status))
        print(f"{mult:<8.2f} {lr_repel:<10.4f} {var:<12.6f} {train_disc:<12.4f} {test_disc:<12.4f} {status} ({elapsed:.1f}s)")
    
    good_range = [r for r in results if r[5] in ["GOOD", "PARTIAL"]]
    if len(good_range) >= 3:
        print(f"\n✓ STABLE across {len(good_range)}/{len(multipliers)} settings")
    else:
        print(f"\n⚠ FRAGILE: only {len(good_range)}/{len(multipliers)} settings work")
    
    return results


def experiment2_scale():
    """Test at larger scale: up to 256-dim, 50+ concepts."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Scale Testing")
    print("=" * 70)
    
    configs = [
        # (feature_dim, shared_dim, n_concepts, instances, epochs)
        (64,  32,  16, 20, 3000),
        (128, 64,  16, 20, 3000),
        (128, 64,  50, 15, 5000),
        (256, 128, 50, 15, 5000),
    ]
    
    print(f"\n{'Config':<30} {'Var':<12} {'Train':<12} {'Test':<12} {'Gen %':<10} {'Time'}")
    print("-" * 80)
    
    for feat_dim, shared_dim, n_concepts, instances, epochs in configs:
        train_vis, train_lang, _, test_vis, test_lang, _ = create_concept_data(
            n_concepts=n_concepts,
            instances_per_concept=instances,
            feature_dim=feat_dim,
        )
        
        atl = ATLSemanticHubV4(
            n_prototypes=max(200, n_concepts * 4),
            feature_dim=feat_dim, shared_dim=shared_dim,
            lr_attract=0.01, lr_repel=0.005, margin_threshold=-0.3,
        ).to(DEVICE)
        
        t0 = time.time()
        for epoch in range(epochs):
            perm = torch.randperm(len(train_vis), device=DEVICE)
            atl.bind_batch(train_vis[perm], train_lang[perm])
        elapsed = time.time() - t0
        
        var = compute_variance_batched(atl, train_vis)
        train_disc = compute_discrimination_batched(atl, train_vis, train_lang)
        test_disc = compute_discrimination_batched(atl, test_vis, test_lang)
        gen_ratio = (test_disc / train_disc * 100) if train_disc > 0 else 0
        
        config_str = f"{feat_dim}→{shared_dim}, {n_concepts}c×{instances}i"
        print(f"{config_str:<30} {var:<12.6f} {train_disc:<12.4f} {test_disc:<12.4f} {gen_ratio:<10.1f} {elapsed:.1f}s")


def experiment3_push_test_disc():
    """
    Push test discrimination to ≥ 0.15.
    
    Strategies: data quantity, noise, LRs, training length, combinations.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Pushing Test Discrimination to ≥ 0.15")
    print("=" * 70)
    
    strategies = [
        # (name, n_concepts, instances, noise, lr_attract, lr_repel, epochs, margin_thresh)
        ("Baseline",       16, 20, 0.5, 0.01,  0.005,  3000, -0.3),
        ("More data",      16, 40, 0.5, 0.01,  0.005,  3000, -0.3),
        ("Less noise",     16, 20, 0.3, 0.01,  0.005,  3000, -0.3),
        ("Higher LR",      16, 20, 0.5, 0.02,  0.01,   3000, -0.3),
        ("Longer train",   16, 20, 0.5, 0.01,  0.005,  8000, -0.3),
        ("Strict margin",  16, 20, 0.5, 0.01,  0.005,  3000, -0.1),
        ("Data+LR",        16, 40, 0.5, 0.02,  0.01,   5000, -0.3),
        ("Combined",       16, 40, 0.3, 0.015, 0.0075, 5000, -0.2),
        ("Combined+long",  16, 40, 0.3, 0.015, 0.0075, 8000, -0.2),
    ]
    
    print(f"\n{'Strategy':<16} {'Train':<10} {'Test':<10} {'Gen %':<10} {'Time':<8} {'Target?'}")
    print("-" * 62)
    
    best_test_disc = 0
    best_strategy = None
    
    for name, n_concepts, instances, noise, lr_a, lr_r, epochs, mt in strategies:
        train_vis, train_lang, _, test_vis, test_lang, _ = create_concept_data(
            n_concepts=n_concepts,
            instances_per_concept=instances,
            feature_dim=64,
            noise_scale=noise,
        )
        
        atl = ATLSemanticHubV4(
            n_prototypes=100, feature_dim=64, shared_dim=32,
            lr_attract=lr_a, lr_repel=lr_r, margin_threshold=mt,
        ).to(DEVICE)
        
        t0 = time.time()
        for epoch in range(epochs):
            perm = torch.randperm(len(train_vis), device=DEVICE)
            atl.bind_batch(train_vis[perm], train_lang[perm])
        elapsed = time.time() - t0
        
        train_disc = compute_discrimination_batched(atl, train_vis, train_lang)
        test_disc = compute_discrimination_batched(atl, test_vis, test_lang)
        gen_ratio = (test_disc / train_disc * 100) if train_disc > 0 else 0
        target = "✓" if test_disc >= 0.15 else ""
        
        print(f"{name:<16} {train_disc:<10.4f} {test_disc:<10.4f} {gen_ratio:<10.1f} {elapsed:<8.1f} {target}")
        
        if test_disc > best_test_disc:
            best_test_disc = test_disc
            best_strategy = name
    
    print(f"\nBest: {best_strategy} with test_disc = {best_test_disc:.4f}")
    
    if best_test_disc >= 0.15:
        print("✓ TARGET MET: test discrimination ≥ 0.15")
    else:
        print(f"✗ TARGET NOT MET: need {0.15 - best_test_disc:.4f} more")


def experiment4_momentum_sensitivity():
    """Test sensitivity to running mean momentum."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Running Mean Momentum Sensitivity")
    print("=" * 70)
    
    momentums = [0.9, 0.95, 0.99, 0.995, 0.999]
    
    train_vis, train_lang, _, test_vis, test_lang, _ = create_concept_data(
        n_concepts=16, instances_per_concept=20, feature_dim=64
    )
    
    print(f"\n{'Momentum':<12} {'Var':<12} {'Train':<12} {'Test':<12}")
    print("-" * 50)
    
    for momentum in momentums:
        atl = ATLSemanticHubV4(
            n_prototypes=100, feature_dim=64, shared_dim=32,
            margin_threshold=-0.3,
        ).to(DEVICE)
        atl.proj_visual.momentum = momentum
        atl.proj_language.momentum = momentum
        atl.proj_audio.momentum = momentum
        
        for epoch in range(3000):
            perm = torch.randperm(len(train_vis), device=DEVICE)
            atl.bind_batch(train_vis[perm], train_lang[perm])
        
        var = compute_variance_batched(atl, train_vis)
        train_disc = compute_discrimination_batched(atl, train_vis, train_lang)
        test_disc = compute_discrimination_batched(atl, test_vis, test_lang)
        
        print(f"{momentum:<12.3f} {var:<12.6f} {train_disc:<12.4f} {test_disc:<12.4f}")


def main():
    print("=" * 70)
    print("ATL V4 SENSITIVITY AND SCALE ANALYSIS")
    print("=" * 70)
    
    experiment1_sensitivity()
    experiment2_scale()
    experiment3_push_test_disc()
    experiment4_momentum_sensitivity()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)


if __name__ == '__main__':
    main()
