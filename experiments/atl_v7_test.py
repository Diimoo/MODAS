#!/usr/bin/env python3
"""
ATL V7 Test: Centered Cross-Covariance Hebbian Learning

V7 replaces V4's running-mean anti-Hebbian with batch centering.
Test: sensitivity, scale, generalization, collapse resistance.
Compare V4 vs V7 head-to-head.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modas.modules.atl_semantic_hub_v4 import ATLSemanticHubV4
from modas.modules.atl_semantic_hub_v7 import ATLSemanticHubV7

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


def create_data(nc, inst, feat, noise=0.5):
    protos = F.normalize(torch.randn(nc, feat, device=DEVICE), dim=1)
    tn = int(inst * 0.75)
    tv, tl, xv, xl = [], [], [], []
    for c in range(nc):
        v = F.normalize(protos[c] + noise * torch.randn(inst, feat, device=DEVICE), dim=1)
        l = F.normalize(protos[c] + noise * torch.randn(inst, feat, device=DEVICE), dim=1)
        p = torch.randperm(inst)
        tv.append(v[p[:tn]]); tl.append(l[p[:tn]])
        xv.append(v[p[tn:]]); xl.append(l[p[tn:]])
    return torch.cat(tv), torch.cat(tl), torch.cat(xv), torch.cat(xl)


def disc(atl, vis, lang, n=30):
    n = min(len(vis), n)
    with torch.no_grad():
        vp = atl.proj_visual(F.normalize(vis[:n], dim=1))
        lp = atl.proj_language(F.normalize(lang[:n], dim=1))
        sim = vp @ lp.T
        m = sim.diag().mean().item()
        mask = ~torch.eye(n, dtype=torch.bool, device=vis.device)
        mm = sim[mask].mean().item()
    return m - mm


def proj_var(atl, vis):
    with torch.no_grad():
        p = atl.proj_visual(F.normalize(vis[:50], dim=1))
        return p.var(dim=0).mean().item()


def train_ep(atl, tv, tl, epochs):
    t0 = time.time()
    for _ in range(epochs):
        p = torch.randperm(len(tv), device=DEVICE)
        atl.bind_batch(tv[p], tl[p])
    return time.time() - t0


def exp1_sensitivity():
    """V7 has only 2 hyperparams: lr and weight_decay. Test both."""
    print("\n" + "=" * 70)
    print("EXP 1: V7 Hyperparameter Sensitivity")
    print("=" * 70)

    tv, tl, xv, xl = create_data(16, 20, 64)

    # Sweep lr
    print(f"\n--- lr sweep (weight_decay=0.001, 3000 epochs) ---")
    print(f"{'lr':<10} {'Var':<12} {'Train':<12} {'Test':<12} {'Status'}")
    print("-" * 56)
    for lr in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        atl = ATLSemanticHubV7(
            n_prototypes=100, feature_dim=64, shared_dim=32,
            lr=lr, weight_decay=0.001, margin_threshold=-0.3,
        ).to(DEVICE)
        train_ep(atl, tv, tl, 3000)
        v = proj_var(atl, tv)
        tr = disc(atl, tv, tl)
        te = disc(atl, xv, xl)
        s = "GOOD" if te >= 0.15 else ("PARTIAL" if te >= 0.05 else ("COLLAPSED" if v < 0.001 else "WEAK"))
        print(f"{lr:<10.3f} {v:<12.6f} {tr:<12.4f} {te:<12.4f} {s}")

    # Sweep weight_decay
    print(f"\n--- weight_decay sweep (lr=0.01, 3000 epochs) ---")
    print(f"{'decay':<10} {'Var':<12} {'Train':<12} {'Test':<12} {'Status'}")
    print("-" * 56)
    for wd in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
        atl = ATLSemanticHubV7(
            n_prototypes=100, feature_dim=64, shared_dim=32,
            lr=0.01, weight_decay=wd, margin_threshold=-0.3,
        ).to(DEVICE)
        train_ep(atl, tv, tl, 3000)
        v = proj_var(atl, tv)
        tr = disc(atl, tv, tl)
        te = disc(atl, xv, xl)
        s = "GOOD" if te >= 0.15 else ("PARTIAL" if te >= 0.05 else ("COLLAPSED" if v < 0.001 else "WEAK"))
        print(f"{wd:<10.4f} {v:<12.6f} {tr:<12.4f} {te:<12.4f} {s}")


def exp2_scale():
    """Head-to-head V4 vs V7 at multiple scales."""
    print("\n" + "=" * 70)
    print("EXP 2: Scale (V4 vs V7)")
    print("=" * 70)

    configs = [
        (64,  32,  16, 20, 3000),
        (128, 64,  16, 20, 3000),
        (128, 64,  50, 15, 5000),
        (256, 128, 50, 15, 5000),
    ]

    print(f"\n{'Config':<25} {'V4 Tr':<10} {'V4 Te':<10} {'V7 Tr':<10} {'V7 Te':<10} {'V7 Var':<12}")
    print("-" * 80)

    for feat, shared, nc, inst, ep in configs:
        tv, tl, xv, xl = create_data(nc, inst, feat)

        atl4 = ATLSemanticHubV4(
            n_prototypes=max(200, nc*4), feature_dim=feat, shared_dim=shared,
            lr_attract=0.01, lr_repel=0.005, margin_threshold=-0.3,
        ).to(DEVICE)
        train_ep(atl4, tv, tl, ep)
        tr4 = disc(atl4, tv, tl)
        te4 = disc(atl4, xv, xl)

        atl7 = ATLSemanticHubV7(
            n_prototypes=max(200, nc*4), feature_dim=feat, shared_dim=shared,
            lr=0.01, weight_decay=0.001, margin_threshold=-0.3,
        ).to(DEVICE)
        train_ep(atl7, tv, tl, ep)
        tr7 = disc(atl7, tv, tl)
        te7 = disc(atl7, xv, xl)
        v7 = proj_var(atl7, tv)

        tag = f"{feat}→{shared}, {nc}c×{inst}i"
        print(f"{tag:<25} {tr4:<10.4f} {te4:<10.4f} {tr7:<10.4f} {te7:<10.4f} {v7:<12.6f}")


def exp3_generalization():
    """Push test disc ≥ 0.15."""
    print("\n" + "=" * 70)
    print("EXP 3: Push Test Disc ≥ 0.15")
    print("=" * 70)

    strategies = [
        # name, feat, shared, nc, inst, noise, lr, wd, epochs, mt
        ("Baseline",     64,  32, 16, 20, 0.5, 0.01,  0.001, 3000, -0.3),
        ("More data",    64,  32, 16, 40, 0.5, 0.01,  0.001, 3000, -0.3),
        ("Less noise",   64,  32, 16, 20, 0.3, 0.01,  0.001, 3000, -0.3),
        ("Higher LR",    64,  32, 16, 20, 0.5, 0.03,  0.001, 3000, -0.3),
        ("Less decay",   64,  32, 16, 20, 0.5, 0.01,  0.0001,3000, -0.3),
        ("Longer",       64,  32, 16, 20, 0.5, 0.01,  0.001, 8000, -0.3),
        ("Combined",     64,  32, 16, 40, 0.3, 0.02,  0.0005,5000, -0.2),
        ("Scale 128",   128,  64, 16, 40, 0.5, 0.01,  0.001, 5000, -0.3),
        ("Scale 256",   256, 128, 16, 40, 0.5, 0.01,  0.001, 5000, -0.3),
    ]

    print(f"\n{'Strategy':<14} {'Train':<10} {'Test':<10} {'Gen%':<10} {'Time':<8} {'✓?'}")
    print("-" * 58)

    best_te, best_name = 0, ""
    for name, feat, shared, nc, inst, noise, lr, wd, ep, mt in strategies:
        tv, tl, xv, xl = create_data(nc, inst, feat, noise)
        atl = ATLSemanticHubV7(
            n_prototypes=100, feature_dim=feat, shared_dim=shared,
            lr=lr, weight_decay=wd, margin_threshold=mt,
        ).to(DEVICE)
        t = train_ep(atl, tv, tl, ep)
        tr = disc(atl, tv, tl)
        te = disc(atl, xv, xl)
        gen = (te / tr * 100) if abs(tr) > 0.01 else 0
        hit = "✓" if te >= 0.15 else ""
        print(f"{name:<14} {tr:<10.4f} {te:<10.4f} {gen:<10.1f} {t:<8.1f} {hit}")
        if te > best_te:
            best_te, best_name = te, name

    print(f"\nBest: {best_name} → test_disc = {best_te:.4f}")
    print(f"{'✓ TARGET MET' if best_te >= 0.15 else '✗ TARGET NOT MET'}")


def exp4_collapse():
    """Long-run collapse test: 10k epochs."""
    print("\n" + "=" * 70)
    print("EXP 4: Collapse Test (10k epochs, V4 vs V7)")
    print("=" * 70)

    tv, tl, xv, xl = create_data(16, 20, 64)

    atl4 = ATLSemanticHubV4(
        n_prototypes=100, feature_dim=64, shared_dim=32,
        lr_attract=0.01, lr_repel=0.005, margin_threshold=-0.3,
    ).to(DEVICE)
    atl7 = ATLSemanticHubV7(
        n_prototypes=100, feature_dim=64, shared_dim=32,
        lr=0.01, weight_decay=0.001, margin_threshold=-0.3,
    ).to(DEVICE)

    print(f"\n{'Epoch':<8} {'V4 Var':<12} {'V4 Te':<10} {'V7 Var':<12} {'V7 Te':<10}")
    print("-" * 55)

    for ep in range(10000):
        p = torch.randperm(len(tv), device=DEVICE)
        atl4.bind_batch(tv[p], tl[p])
        atl7.bind_batch(tv[p], tl[p])
        if (ep + 1) % 1000 == 0:
            v4v = proj_var(atl4, tv)
            v4t = disc(atl4, xv, xl)
            v7v = proj_var(atl7, tv)
            v7t = disc(atl7, xv, xl)
            print(f"{ep+1:<8} {v4v:<12.6f} {v4t:<10.4f} {v7v:<12.6f} {v7t:<10.4f}")


def main():
    print("=" * 70)
    print("ATL V7: CENTERED CROSS-COVARIANCE HEBBIAN")
    print("=" * 70)
    exp1_sensitivity()
    exp2_scale()
    exp3_generalization()
    exp4_collapse()
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
