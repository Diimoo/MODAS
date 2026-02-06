#!/usr/bin/env python3
"""
Structured Binding Test: The Developmental Input Hypothesis

Previous experiments (V3-V7) asked: "Can Hebbian learning align arbitrary spaces?"
Answer: No. Insufficient SNR for projection learning.

This experiment asks the REAL question:
"Given developmentally structured input where each modality has internal
concept-level clustering, can Hebbian ASSOCIATION bind cross-modal regularities?"

Key difference from all prior experiments:
- Visual and language features are in INDEPENDENT geometric spaces
  (no shared prototype — different random bases per modality)
- BUT: within each space, same-concept instances cluster together
  (simulating LCA learning category-correlated bases)
- The binding mechanism must learn "vis cluster A ↔ lang cluster B"
  purely from co-occurrence, not geometric alignment

This tests V6 (associative) vs V4 (projection) to confirm that:
- V4 CANNOT work (no linear mapping exists between independent spaces)
- V6 CAN work IF competitive layers form concept-level clusters

Three conditions tested:
A. Shared geometry (prior experiments): same prototype, same noise
B. Independent geometry: separate prototypes, concept clustering
C. No structure: random features, no concept clustering
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modas.modules.atl_semantic_hub_v4 import ATLSemanticHubV4
from modas.modules.atl_semantic_hub_v6 import ATLSemanticHubV6

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ============================================================
# Data generation: three regimes
# ============================================================

def create_shared_geometry(nc, inst, feat, noise=0.5):
    """
    Condition A: Shared prototype (what V3-V7 experiments used).
    Visual and language share the same concept prototype.
    A linear projection COULD align them.
    """
    protos = F.normalize(torch.randn(nc, feat, device=DEVICE), dim=1)
    tn = int(inst * 0.75)
    tv, tl, xv, xl, tc, xc = [], [], [], [], [], []
    for c in range(nc):
        v = F.normalize(protos[c] + noise * torch.randn(inst, feat, device=DEVICE), dim=1)
        l = F.normalize(protos[c] + noise * torch.randn(inst, feat, device=DEVICE), dim=1)
        p = torch.randperm(inst)
        tv.append(v[p[:tn]]); tl.append(l[p[:tn]])
        xv.append(v[p[tn:]]); xl.append(l[p[tn:]])
        tc.extend([c] * tn); xc.extend([c] * (inst - tn))
    return (torch.cat(tv), torch.cat(tl), torch.cat(xv), torch.cat(xl),
            torch.tensor(tc, device=DEVICE), torch.tensor(xc, device=DEVICE))


def create_independent_geometry(nc, inst, feat, noise=0.5):
    """
    Condition B: Independent concept prototypes per modality.
    Visual and language live in geometrically unrelated spaces.
    NO linear projection can align them.
    But each space has concept-level clustering.
    
    This simulates: LCA learns visual bases that cluster by category,
    Word2Vec has its own semantic structure.
    """
    # Separate prototypes for each modality
    vis_protos = F.normalize(torch.randn(nc, feat, device=DEVICE), dim=1)
    lang_protos = F.normalize(torch.randn(nc, feat, device=DEVICE), dim=1)
    
    tn = int(inst * 0.75)
    tv, tl, xv, xl, tc, xc = [], [], [], [], [], []
    for c in range(nc):
        v = F.normalize(vis_protos[c] + noise * torch.randn(inst, feat, device=DEVICE), dim=1)
        l = F.normalize(lang_protos[c] + noise * torch.randn(inst, feat, device=DEVICE), dim=1)
        p = torch.randperm(inst)
        tv.append(v[p[:tn]]); tl.append(l[p[:tn]])
        xv.append(v[p[tn:]]); xl.append(l[p[tn:]])
        tc.extend([c] * tn); xc.extend([c] * (inst - tn))
    return (torch.cat(tv), torch.cat(tl), torch.cat(xv), torch.cat(xl),
            torch.tensor(tc, device=DEVICE), torch.tensor(xc, device=DEVICE))


def create_no_structure(nc, inst, feat):
    """
    Condition C: No concept structure at all.
    Random features, no clustering. Control condition.
    Nothing should work here.
    """
    tn = int(inst * 0.75)
    total_train = nc * tn
    total_test = nc * (inst - tn)
    tv = F.normalize(torch.randn(total_train, feat, device=DEVICE), dim=1)
    tl = F.normalize(torch.randn(total_train, feat, device=DEVICE), dim=1)
    xv = F.normalize(torch.randn(total_test, feat, device=DEVICE), dim=1)
    xl = F.normalize(torch.randn(total_test, feat, device=DEVICE), dim=1)
    tc = torch.arange(nc, device=DEVICE).repeat_interleave(tn)
    xc = torch.arange(nc, device=DEVICE).repeat_interleave(inst - tn)
    return tv, tl, xv, xl, tc, xc


# ============================================================
# Evaluation
# ============================================================

def disc_v4(atl, vis, lang, n=30):
    """Discrimination via projection similarity (V4)."""
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
    """Discrimination via associative binding (V6)."""
    n = min(len(vis), n)
    with torch.no_grad():
        va = atl.vis_layer(F.normalize(vis[:n], dim=1), k=atl.k)
        la = atl.lang_layer(F.normalize(lang[:n], dim=1), k=atl.k)
        sim = va @ atl.W_vl @ la.T
        m = sim.diag().mean().item()
        mask = ~torch.eye(n, dtype=torch.bool, device=vis.device)
        mm = sim[mask].mean().item()
    return m - mm


def cluster_quality(layer, features, concepts, k=3):
    """
    Measure how well a competitive layer clusters by concept.
    Returns within-concept activation similarity vs between-concept.
    """
    with torch.no_grad():
        acts = layer(F.normalize(features, dim=1), k=k)
        ucs = concepts.unique()
        within, between = [], []
        for c in ucs:
            mask_c = concepts == c
            if mask_c.sum() < 2:
                continue
            acts_c = acts[mask_c]
            # Within: pairwise similarity of activation patterns for same concept
            sim_c = acts_c @ acts_c.T
            n = len(acts_c)
            mask_offdiag = ~torch.eye(n, dtype=torch.bool, device=features.device)
            within.append(sim_c[mask_offdiag].mean().item())
            # Between: similarity with other concepts
            acts_other = acts[~mask_c]
            if len(acts_other) > 0:
                idx = torch.randperm(len(acts_other))[:min(20, len(acts_other))]
                sim_b = acts_c @ acts_other[idx].T
                between.append(sim_b.mean().item())
        w = np.mean(within) if within else 0
        b = np.mean(between) if between else 0
        return w, b, w - b


def train_ep(atl, tv, tl, epochs):
    t0 = time.time()
    for _ in range(epochs):
        p = torch.randperm(len(tv), device=DEVICE)
        atl.bind_batch(tv[p], tl[p])
    return time.time() - t0


# ============================================================
# Main experiment
# ============================================================

def run_condition(name, tv, tl, xv, xl, tc, xc, feat, nc, epochs=3000):
    """Run V4 and V6 on a given data condition."""
    shared = feat // 2
    
    # V4 (projection-based)
    atl4 = ATLSemanticHubV4(
        n_prototypes=200, feature_dim=feat, shared_dim=shared,
        lr_attract=0.01, lr_repel=0.005, margin_threshold=-0.3,
    ).to(DEVICE)
    t4 = train_ep(atl4, tv, tl, epochs)
    tr4 = disc_v4(atl4, tv, tl)
    te4 = disc_v4(atl4, xv, xl)

    # V6 (associative binding)
    # Use more prototypes: 3-4x concept count
    n_protos = max(50, nc * 4)
    atl6 = ATLSemanticHubV6(
        feature_dim=feat, n_vis_protos=n_protos, n_lang_protos=n_protos,
        lr_bind=0.1, lr_unbind=0.02, lr_proto=0.05, k=3,
    ).to(DEVICE)
    t6 = train_ep(atl6, tv, tl, epochs)
    tr6 = disc_v6(atl6, tv, tl)
    te6 = disc_v6(atl6, xv, xl)
    
    # Cluster quality for V6
    cq_vis_w, cq_vis_b, cq_vis = cluster_quality(atl6.vis_layer, tv, tc, k=3)
    cq_lang_w, cq_lang_b, cq_lang = cluster_quality(atl6.lang_layer, tl, tc, k=3)
    
    stats6 = atl6.get_stats()
    
    return {
        'name': name,
        'v4_train': tr4, 'v4_test': te4, 'v4_time': t4,
        'v6_train': tr6, 'v6_test': te6, 'v6_time': t6,
        'vis_cluster': cq_vis, 'lang_cluster': cq_lang,
        'vis_w': cq_vis_w, 'vis_b': cq_vis_b,
        'lang_w': cq_lang_w, 'lang_b': cq_lang_b,
        'bind_sparsity': stats6['bind_sparsity'],
        'bind_max': stats6['bind_max'],
    }


def print_result(r):
    print(f"\n--- {r['name']} ---")
    print(f"  V4 projection:  train={r['v4_train']:.4f}  test={r['v4_test']:.4f}  ({r['v4_time']:.1f}s)")
    print(f"  V6 association:  train={r['v6_train']:.4f}  test={r['v6_test']:.4f}  ({r['v6_time']:.1f}s)")
    print(f"  V6 vis cluster:  within={r['vis_w']:.4f}  between={r['vis_b']:.4f}  gap={r['vis_cluster']:.4f}")
    print(f"  V6 lang cluster: within={r['lang_w']:.4f}  between={r['lang_b']:.4f}  gap={r['lang_cluster']:.4f}")
    print(f"  V6 binding:      sparsity={r['bind_sparsity']:.4f}  max={r['bind_max']:.4f}")


def exp1_three_conditions():
    """Compare V4 vs V6 under all three data conditions."""
    print("\n" + "=" * 70)
    print("EXP 1: Three Conditions (64-dim, 16 concepts)")
    print("=" * 70)
    
    nc, inst, feat = 16, 20, 64
    
    tv_a, tl_a, xv_a, xl_a, tc_a, xc_a = create_shared_geometry(nc, inst, feat)
    tv_b, tl_b, xv_b, xl_b, tc_b, xc_b = create_independent_geometry(nc, inst, feat)
    tv_c, tl_c, xv_c, xl_c, tc_c, xc_c = create_no_structure(nc, inst, feat)
    
    results = []
    results.append(run_condition("A: Shared geometry", tv_a, tl_a, xv_a, xl_a, tc_a, xc_a, feat, nc))
    results.append(run_condition("B: Independent geometry", tv_b, tl_b, xv_b, xl_b, tc_b, xc_b, feat, nc))
    results.append(run_condition("C: No structure", tv_c, tl_c, xv_c, xl_c, tc_c, xc_c, feat, nc))
    
    for r in results:
        print_result(r)
    
    print("\n--- Summary ---")
    print(f"{'Condition':<30} {'V4 Test':<12} {'V6 Test':<12} {'Vis Gap':<12} {'Lang Gap':<12}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<30} {r['v4_test']:<12.4f} {r['v6_test']:<12.4f} {r['vis_cluster']:<12.4f} {r['lang_cluster']:<12.4f}")


def exp2_scale():
    """Scale test under independent geometry (the realistic condition)."""
    print("\n" + "=" * 70)
    print("EXP 2: Scale Under Independent Geometry")
    print("=" * 70)
    
    configs = [
        # feat, nc, inst, epochs
        (64,   16, 20, 3000),
        (64,   16, 40, 3000),
        (128,  16, 20, 5000),
        (128,  50, 15, 5000),
        (256,  16, 40, 5000),
        (256,  50, 15, 8000),
    ]
    
    print(f"\n{'Config':<25} {'V4 Test':<10} {'V6 Test':<10} {'V6 Train':<10} {'Vis Gap':<10} {'Lang Gap'}")
    print("-" * 75)
    
    for feat, nc, inst, ep in configs:
        tv, tl, xv, xl, tc, xc = create_independent_geometry(nc, inst, feat)
        r = run_condition(f"{feat}d, {nc}c×{inst}i", tv, tl, xv, xl, tc, xc, feat, nc, ep)
        tag = f"{feat}d, {nc}c×{inst}i"
        print(f"{tag:<25} {r['v4_test']:<10.4f} {r['v6_test']:<10.4f} {r['v6_train']:<10.4f} {r['vis_cluster']:<10.4f} {r['lang_cluster']:.4f}")


def exp3_noise_sweep():
    """How does input noise affect V6's cluster quality and discrimination?"""
    print("\n" + "=" * 70)
    print("EXP 3: Noise Sweep (Independent Geometry, 64-dim, 16 concepts)")
    print("=" * 70)
    
    nc, inst, feat = 16, 20, 64
    
    print(f"\n{'Noise':<8} {'V6 Train':<10} {'V6 Test':<10} {'Vis Gap':<10} {'Lang Gap':<10} {'Bind Sp'}")
    print("-" * 60)
    
    for noise in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        tv, tl, xv, xl, tc, xc = create_independent_geometry(nc, inst, feat, noise)
        r = run_condition(f"noise={noise}", tv, tl, xv, xl, tc, xc, feat, nc, 3000)
        print(f"{noise:<8.1f} {r['v6_train']:<10.4f} {r['v6_test']:<10.4f} {r['vis_cluster']:<10.4f} {r['lang_cluster']:<10.4f} {r['bind_sparsity']:.4f}")


def exp4_k_sweep():
    """Sweep top-k sparsity for V6 under independent geometry."""
    print("\n" + "=" * 70)
    print("EXP 4: top-k Sweep (Independent Geometry, 64-dim, 16 concepts)")
    print("=" * 70)
    
    nc, inst, feat = 16, 20, 64
    tv, tl, xv, xl, tc, xc = create_independent_geometry(nc, inst, feat, noise=0.3)
    
    print(f"\n{'k':<6} {'Train':<10} {'Test':<10} {'Vis Gap':<10} {'Lang Gap':<10} {'Bind Sp'}")
    print("-" * 55)
    
    for k in [1, 2, 3, 5, 8, 10]:
        n_protos = nc * 4
        atl6 = ATLSemanticHubV6(
            feature_dim=feat, n_vis_protos=n_protos, n_lang_protos=n_protos,
            lr_bind=0.1, lr_unbind=0.02, lr_proto=0.05, k=k,
        ).to(DEVICE)
        train_ep(atl6, tv, tl, 3000)
        tr = disc_v6(atl6, tv, tl)
        te = disc_v6(atl6, xv, xl)
        cq_vis_w, cq_vis_b, cq_vis = cluster_quality(atl6.vis_layer, tv, tc, k=k)
        cq_lang_w, cq_lang_b, cq_lang = cluster_quality(atl6.lang_layer, tl, tc, k=k)
        stats = atl6.get_stats()
        print(f"{k:<6} {tr:<10.4f} {te:<10.4f} {cq_vis:<10.4f} {cq_lang:<10.4f} {stats['bind_sparsity']:.4f}")


def exp5_proto_ratio():
    """Sweep prototype count relative to concept count."""
    print("\n" + "=" * 70)
    print("EXP 5: Prototype Ratio (Independent Geometry, 64-dim, 16 concepts)")
    print("=" * 70)
    
    nc, inst, feat = 16, 20, 64
    tv, tl, xv, xl, tc, xc = create_independent_geometry(nc, inst, feat, noise=0.3)
    
    print(f"\n{'Protos':<8} {'Ratio':<8} {'Train':<10} {'Test':<10} {'Vis Gap':<10} {'Bind Sp'}")
    print("-" * 55)
    
    for np_ in [16, 32, 48, 64, 100, 200]:
        atl6 = ATLSemanticHubV6(
            feature_dim=feat, n_vis_protos=np_, n_lang_protos=np_,
            lr_bind=0.1, lr_unbind=0.02, lr_proto=0.05, k=3,
        ).to(DEVICE)
        train_ep(atl6, tv, tl, 3000)
        tr = disc_v6(atl6, tv, tl)
        te = disc_v6(atl6, xv, xl)
        cq_vis_w, cq_vis_b, cq_vis = cluster_quality(atl6.vis_layer, tv, tc, k=3)
        stats = atl6.get_stats()
        print(f"{np_:<8} {np_/nc:<8.1f} {tr:<10.4f} {te:<10.4f} {cq_vis:<10.4f} {stats['bind_sparsity']:.4f}")


def main():
    print("=" * 70)
    print("STRUCTURED BINDING TEST: DEVELOPMENTAL INPUT HYPOTHESIS")
    print("=" * 70)
    exp1_three_conditions()
    exp2_scale()
    exp3_noise_sweep()
    exp4_k_sweep()
    exp5_proto_ratio()
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
