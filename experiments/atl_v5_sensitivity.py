#!/usr/bin/env python3
"""
ATL V5 Sensitivity and Scale Analysis (GPU-accelerated)

V5 fixes: bottleneck projection + covariance decorrelation.
Compare against V4 on the same tests that V4 failed.
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
from modas.modules.atl_semantic_hub_v5 import ATLSemanticHubV5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def create_concept_data(
    n_concepts: int,
    instances_per_concept: int,
    feature_dim: int,
    noise_scale: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create train/test tensors on DEVICE."""
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


def compute_disc(atl, vis, lang, n_eval=30):
    """Batched discrimination."""
    n = min(len(vis), n_eval)
    v, l = vis[:n], lang[:n]
    with torch.no_grad():
        if hasattr(atl, 'proj_visual'):
            v_proj = atl.proj_visual(F.normalize(v, dim=1))
            l_proj = atl.proj_language(F.normalize(l, dim=1))
        else:
            v_proj = atl.project(F.normalize(v, dim=1), 'visual')
            l_proj = atl.project(F.normalize(l, dim=1), 'language')
        sim = v_proj @ l_proj.T
        matched = sim.diag().mean().item()
        mask = ~torch.eye(n, dtype=torch.bool, device=vis.device)
        mismatched = sim[mask].mean().item()
    return matched - mismatched


def compute_var(atl, vis):
    """Projection variance."""
    with torch.no_grad():
        projs = atl.proj_visual(F.normalize(vis[:50], dim=1))
        return projs.var(dim=0).mean().item()


def train_model(atl, train_vis, train_lang, epochs):
    """Train for N epochs, return time."""
    t0 = time.time()
    for _ in range(epochs):
        perm = torch.randperm(len(train_vis), device=DEVICE)
        atl.bind_batch(train_vis[perm], train_lang[perm])
    return time.time() - t0


# ============================================================
# EXPERIMENT 1: Sensitivity (lr_decor for V5, lr_repel for V4)
# ============================================================
def experiment1():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Decorrelation Strength Sensitivity")
    print("=" * 70)

    train_vis, train_lang, _, test_vis, test_lang, _ = create_concept_data(
        n_concepts=16, instances_per_concept=20, feature_dim=64
    )

    mults = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
    base = 0.005

    print(f"\nBase lr_decor = {base}, 3000 epochs, 16 concepts, 64-dim")
    print(f"\n{'Mult':<8} {'Var':<12} {'Train':<12} {'Test':<12} {'Status':<12} {'Time'}")
    print("-" * 64)

    results = []
    for m in mults:
        atl = ATLSemanticHubV5(
            n_prototypes=100, feature_dim=64, bottleneck_dim=16,
            shared_dim=32, lr=0.01, lr_decor=base * m, margin_threshold=-0.3,
        ).to(DEVICE)

        elapsed = train_model(atl, train_vis, train_lang, 3000)
        var = compute_var(atl, train_vis)
        tr = compute_disc(atl, train_vis, train_lang)
        te = compute_disc(atl, test_vis, test_lang)

        status = "COLLAPSED" if var < 0.001 else ("GOOD" if te >= 0.15 else ("PARTIAL" if te >= 0.05 else "WEAK"))
        results.append((m, var, tr, te, status))
        print(f"{m:<8.2f} {var:<12.6f} {tr:<12.4f} {te:<12.4f} {status:<12} {elapsed:.1f}s")

    good = sum(1 for r in results if r[4] in ["GOOD", "PARTIAL"])
    print(f"\n{'✓ STABLE' if good >= 3 else '⚠ FRAGILE'}: {good}/{len(mults)} settings work")


# ============================================================
# EXPERIMENT 2: Scale
# ============================================================
def experiment2():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Scale Testing (V4 vs V5)")
    print("=" * 70)

    configs = [
        (64,  32, 16, 16, 20, 3000),
        (128, 64, 16, 24, 20, 3000),
        (128, 64, 32, 24, 15, 5000),
        (256, 128, 32, 32, 15, 5000),
    ]

    print(f"\n{'Config':<25} {'V4 Test':<12} {'V5 Test':<12} {'V5 Var':<12} {'V5 Gen%':<10} {'Time'}")
    print("-" * 75)

    for feat, shared, bn, n_concepts, inst, epochs in configs:
        train_vis, train_lang, _, test_vis, test_lang, _ = create_concept_data(
            n_concepts=n_concepts, instances_per_concept=inst, feature_dim=feat,
        )

        # V4
        atl4 = ATLSemanticHubV4(
            n_prototypes=max(200, n_concepts * 4),
            feature_dim=feat, shared_dim=shared,
            lr_attract=0.01, lr_repel=0.005, margin_threshold=-0.3,
        ).to(DEVICE)
        train_model(atl4, train_vis, train_lang, epochs)
        te4 = compute_disc(atl4, test_vis, test_lang)

        # V5
        atl5 = ATLSemanticHubV5(
            n_prototypes=max(200, n_concepts * 4),
            feature_dim=feat, bottleneck_dim=bn, shared_dim=shared,
            lr=0.01, lr_decor=0.005, margin_threshold=-0.3,
        ).to(DEVICE)
        t = train_model(atl5, train_vis, train_lang, epochs)
        var5 = compute_var(atl5, train_vis)
        tr5 = compute_disc(atl5, train_vis, train_lang)
        te5 = compute_disc(atl5, test_vis, test_lang)
        gen = (te5 / tr5 * 100) if tr5 > 0.01 else 0

        tag = f"{feat}→{bn}→{shared}, {n_concepts}c"
        print(f"{tag:<25} {te4:<12.4f} {te5:<12.4f} {var5:<12.6f} {gen:<10.1f} {t:.1f}s")


# ============================================================
# EXPERIMENT 3: Push test disc ≥ 0.15
# ============================================================
def experiment3():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Push Test Discrimination ≥ 0.15 (V5)")
    print("=" * 70)

    strategies = [
        # name, feat, bn, shared, concepts, inst, noise, lr, lr_d, epochs, mt
        ("Baseline",     64, 16, 32, 16, 20, 0.5, 0.01,  0.005,  3000, -0.3),
        ("More data",    64, 16, 32, 16, 40, 0.5, 0.01,  0.005,  3000, -0.3),
        ("Less noise",   64, 16, 32, 16, 20, 0.3, 0.01,  0.005,  3000, -0.3),
        ("Higher LR",    64, 16, 32, 16, 20, 0.5, 0.02,  0.01,   3000, -0.3),
        ("Longer",       64, 16, 32, 16, 20, 0.5, 0.01,  0.005,  8000, -0.3),
        ("Bigger bn",    64, 32, 32, 16, 20, 0.5, 0.01,  0.005,  3000, -0.3),
        ("Combined",     64, 16, 32, 16, 40, 0.3, 0.015, 0.0075, 5000, -0.2),
        ("Full scale",  128, 24, 64, 16, 40, 0.5, 0.01,  0.005,  5000, -0.3),
    ]

    print(f"\n{'Strategy':<14} {'Train':<10} {'Test':<10} {'Gen%':<10} {'Time':<8} {'✓?'}")
    print("-" * 58)

    best_te, best_name = 0, ""
    for name, feat, bn, shared, nc, inst, noise, lr, lr_d, ep, mt in strategies:
        train_vis, train_lang, _, test_vis, test_lang, _ = create_concept_data(
            n_concepts=nc, instances_per_concept=inst, feature_dim=feat, noise_scale=noise,
        )
        atl = ATLSemanticHubV5(
            n_prototypes=100, feature_dim=feat, bottleneck_dim=bn,
            shared_dim=shared, lr=lr, lr_decor=lr_d, margin_threshold=mt,
        ).to(DEVICE)
        t = train_model(atl, train_vis, train_lang, ep)
        tr = compute_disc(atl, train_vis, train_lang)
        te = compute_disc(atl, test_vis, test_lang)
        gen = (te / tr * 100) if tr > 0.01 else 0
        hit = "✓" if te >= 0.15 else ""
        print(f"{name:<14} {tr:<10.4f} {te:<10.4f} {gen:<10.1f} {t:<8.1f} {hit}")
        if te > best_te:
            best_te, best_name = te, name

    print(f"\nBest: {best_name} → test_disc = {best_te:.4f}")
    print(f"{'✓ TARGET MET' if best_te >= 0.15 else '✗ TARGET NOT MET'}")


# ============================================================
# EXPERIMENT 4: Bottleneck dimension sweep
# ============================================================
def experiment4():
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Bottleneck Dimension Sweep")
    print("=" * 70)

    train_vis, train_lang, _, test_vis, test_lang, _ = create_concept_data(
        n_concepts=16, instances_per_concept=20, feature_dim=128
    )

    bns = [4, 8, 16, 24, 32, 48, 64]
    print(f"\n128-dim input → bn → 64-dim shared, 3000 epochs")
    print(f"\n{'bn_dim':<10} {'Params':<12} {'Var':<12} {'Train':<12} {'Test':<12}")
    print("-" * 60)

    for bn in bns:
        params = 128 * bn + bn * 64  # W1 + W2 parameter count
        atl = ATLSemanticHubV5(
            n_prototypes=100, feature_dim=128, bottleneck_dim=bn,
            shared_dim=64, lr=0.01, lr_decor=0.005, margin_threshold=-0.3,
        ).to(DEVICE)
        train_model(atl, train_vis, train_lang, 3000)
        var = compute_var(atl, train_vis)
        tr = compute_disc(atl, train_vis, train_lang)
        te = compute_disc(atl, test_vis, test_lang)
        print(f"{bn:<10} {params:<12} {var:<12.6f} {tr:<12.4f} {te:<12.4f}")


def main():
    print("=" * 70)
    print("ATL V5: BOTTLENECK + COVARIANCE DECORRELATION")
    print("=" * 70)

    experiment1()
    experiment2()
    experiment3()
    experiment4()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
