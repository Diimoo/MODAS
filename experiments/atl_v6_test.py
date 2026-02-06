#!/usr/bin/env python3
"""
ATL V6 Test: Heteroassociative Hebbian Binding

Compare V6 against V4 on all the tests V4 failed:
1. Sensitivity to hyperparameters
2. Scale (128+ dim, 50+ concepts)
3. Generalization (test disc ≥ 0.15 on held-out instances)
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
from modas.modules.atl_semantic_hub_v6 import ATLSemanticHubV6

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def create_data(n_concepts, inst, feat_dim, noise=0.5):
    """Create train/test on DEVICE."""
    protos = F.normalize(torch.randn(n_concepts, feat_dim, device=DEVICE), dim=1)
    train_n = int(inst * 0.75)
    test_n = inst - train_n
    tv, tl, xv, xl = [], [], [], []
    for c in range(n_concepts):
        v = F.normalize(protos[c] + noise * torch.randn(inst, feat_dim, device=DEVICE), dim=1)
        l = F.normalize(protos[c] + noise * torch.randn(inst, feat_dim, device=DEVICE), dim=1)
        p = torch.randperm(inst)
        tv.append(v[p[:train_n]]); tl.append(l[p[:train_n]])
        xv.append(v[p[train_n:]]); xl.append(l[p[train_n:]])
    return torch.cat(tv), torch.cat(tl), torch.cat(xv), torch.cat(xl)


def disc_v4(atl, vis, lang, n=30):
    """Discrimination for V4 (projection-based)."""
    n = min(len(vis), n)
    with torch.no_grad():
        vp = atl.proj_visual(F.normalize(vis[:n], dim=1))
        lp = atl.proj_language(F.normalize(lang[:n], dim=1))
        sim = vp @ lp.T
        m = sim.diag().mean().item()
        mask = ~torch.eye(n, dtype=torch.bool, device=vis.device)
        mm = sim[mask].mean().item()
    return m - mm


def disc_v6(atl, vis, lang, n=30):
    """Discrimination for V6 (association-based)."""
    n = min(len(vis), n)
    with torch.no_grad():
        va = atl.vis_layer(F.normalize(vis[:n], dim=1))   # (n, P_v)
        la = atl.lang_layer(F.normalize(lang[:n], dim=1))  # (n, P_l)
        # Similarity through binding matrix
        sim = va @ atl.W_vl @ la.T  # (n, n)
        m = sim.diag().mean().item()
        mask = ~torch.eye(n, dtype=torch.bool, device=vis.device)
        mm = sim[mask].mean().item()
    return m - mm


def train(atl, tv, tl, epochs):
    t0 = time.time()
    for _ in range(epochs):
        p = torch.randperm(len(tv), device=DEVICE)
        atl.bind_batch(tv[p], tl[p])
    return time.time() - t0


# ============================================================
def exp1_sensitivity():
    print("\n" + "=" * 70)
    print("EXP 1: lr_bind Sensitivity (V6)")
    print("=" * 70)

    tv, tl, xv, xl = create_data(16, 20, 64)
    base = 0.1
    mults = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]

    print(f"\n{'Mult':<8} {'lr_bind':<10} {'Train':<12} {'Test':<12} {'Bind Sp':<12} {'Status'}")
    print("-" * 66)

    results = []
    for m in mults:
        atl = ATLSemanticHubV6(
            feature_dim=64, n_vis_protos=50, n_lang_protos=50,
            lr_bind=base * m, lr_unbind=0.02 * m, lr_proto=0.05,
        ).to(DEVICE)
        t = train(atl, tv, tl, 3000)
        tr = disc_v6(atl, tv, tl)
        te = disc_v6(atl, xv, xl)
        stats = atl.get_stats()
        status = "GOOD" if te >= 0.15 else ("PARTIAL" if te >= 0.05 else "WEAK")
        results.append((m, tr, te, status))
        print(f"{m:<8.2f} {base*m:<10.3f} {tr:<12.4f} {te:<12.4f} {stats['bind_sparsity']:<12.4f} {status} ({t:.1f}s)")

    good = sum(1 for r in results if r[3] in ["GOOD", "PARTIAL"])
    print(f"\n{'✓ STABLE' if good >= 3 else '⚠ FRAGILE'}: {good}/{len(mults)} settings work")


# ============================================================
def exp2_scale():
    print("\n" + "=" * 70)
    print("EXP 2: Scale (V4 vs V6)")
    print("=" * 70)

    configs = [
        # feat, n_concepts, inst, epochs, v_protos, l_protos
        (64,  16, 20, 3000, 50,  50),
        (128, 16, 20, 3000, 50,  50),
        (128, 50, 15, 5000, 100, 100),
        (256, 50, 15, 5000, 100, 100),
    ]

    print(f"\n{'Config':<25} {'V4 Test':<12} {'V6 Test':<12} {'V6 Train':<12} {'Time'}")
    print("-" * 65)

    for feat, nc, inst, ep, vp, lp in configs:
        tv, tl, xv, xl = create_data(nc, inst, feat)

        # V4
        atl4 = ATLSemanticHubV4(
            n_prototypes=max(200, nc * 4), feature_dim=feat, shared_dim=feat // 2,
            lr_attract=0.01, lr_repel=0.005, margin_threshold=-0.3,
        ).to(DEVICE)
        train(atl4, tv, tl, ep)
        te4 = disc_v4(atl4, xv, xl)

        # V6
        atl6 = ATLSemanticHubV6(
            feature_dim=feat, n_vis_protos=vp, n_lang_protos=lp,
            lr_bind=0.1, lr_unbind=0.02, lr_proto=0.05,
        ).to(DEVICE)
        t = train(atl6, tv, tl, ep)
        tr6 = disc_v6(atl6, tv, tl)
        te6 = disc_v6(atl6, xv, xl)

        tag = f"{feat}d, {nc}c×{inst}i"
        print(f"{tag:<25} {te4:<12.4f} {te6:<12.4f} {tr6:<12.4f} {t:.1f}s")


# ============================================================
def exp3_generalization():
    print("\n" + "=" * 70)
    print("EXP 3: Push Test Disc ≥ 0.15 (V6)")
    print("=" * 70)

    strategies = [
        # name, feat, nc, inst, noise, lr_b, lr_u, lr_p, vp, lp, epochs
        ("Baseline",    64,  16, 20, 0.5, 0.1,  0.02, 0.05, 50,  50,  3000),
        ("More data",   64,  16, 40, 0.5, 0.1,  0.02, 0.05, 50,  50,  3000),
        ("Less noise",  64,  16, 20, 0.3, 0.1,  0.02, 0.05, 50,  50,  3000),
        ("More protos", 64,  16, 20, 0.5, 0.1,  0.02, 0.05, 100, 100, 3000),
        ("Longer",      64,  16, 20, 0.5, 0.1,  0.02, 0.05, 50,  50,  8000),
        ("Higher LR",   64,  16, 20, 0.5, 0.2,  0.04, 0.05, 50,  50,  3000),
        ("Combined",    64,  16, 40, 0.3, 0.15, 0.03, 0.05, 100, 100, 5000),
        ("Scale test", 128,  16, 40, 0.5, 0.1,  0.02, 0.05, 100, 100, 5000),
    ]

    print(f"\n{'Strategy':<14} {'Train':<10} {'Test':<10} {'Gen%':<10} {'Time':<8} {'✓?'}")
    print("-" * 58)

    best_te, best_name = 0, ""
    for name, feat, nc, inst, noise, lr_b, lr_u, lr_p, vp, lp, ep in strategies:
        tv, tl, xv, xl = create_data(nc, inst, feat, noise)
        atl = ATLSemanticHubV6(
            feature_dim=feat, n_vis_protos=vp, n_lang_protos=lp,
            lr_bind=lr_b, lr_unbind=lr_u, lr_proto=lr_p,
        ).to(DEVICE)
        t = train(atl, tv, tl, ep)
        tr = disc_v6(atl, tv, tl)
        te = disc_v6(atl, xv, xl)
        gen = (te / tr * 100) if tr > 0.01 else 0
        hit = "✓" if te >= 0.15 else ""
        print(f"{name:<14} {tr:<10.4f} {te:<10.4f} {gen:<10.1f} {t:<8.1f} {hit}")
        if te > best_te:
            best_te, best_name = te, name

    print(f"\nBest: {best_name} → test_disc = {best_te:.4f}")
    print(f"{'✓ TARGET MET' if best_te >= 0.15 else '✗ TARGET NOT MET'}")


# ============================================================
def exp4_long_run():
    """Test stability over 10k epochs."""
    print("\n" + "=" * 70)
    print("EXP 4: Long-Run Stability (10k epochs)")
    print("=" * 70)

    tv, tl, xv, xl = create_data(16, 20, 64)
    atl = ATLSemanticHubV6(
        feature_dim=64, n_vis_protos=50, n_lang_protos=50,
        lr_bind=0.1, lr_unbind=0.02, lr_proto=0.05,
    ).to(DEVICE)

    print(f"\n{'Epoch':<10} {'Train':<12} {'Test':<12} {'Bind Max':<12} {'Bind Sp':<12}")
    print("-" * 60)

    for ep in range(10000):
        p = torch.randperm(len(tv), device=DEVICE)
        atl.bind_batch(tv[p], tl[p])
        if (ep + 1) % 1000 == 0:
            tr = disc_v6(atl, tv, tl)
            te = disc_v6(atl, xv, xl)
            s = atl.get_stats()
            print(f"{ep+1:<10} {tr:<12.4f} {te:<12.4f} {s['bind_max']:<12.4f} {s['bind_sparsity']:<12.4f}")


def main():
    print("=" * 70)
    print("ATL V6: HETEROASSOCIATIVE HEBBIAN BINDING")
    print("=" * 70)
    exp1_sensitivity()
    exp2_scale()
    exp3_generalization()
    exp4_long_run()
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
