#!/usr/bin/env python3
"""
V6 Improved: Push noise tolerance beyond 0.2

The structured binding test showed V6 works at noise≤0.2 but fails at 0.3+.
Real LCA codes probably have higher effective noise, so we need to push
the threshold up.

Improvements to test:
1. Two-phase training: cluster first (unsupervised), then bind
2. More training data per concept
3. Warm-start competitive layer with more epochs
4. Adjusted k and prototype counts
5. Scale to 128-dim
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modas.modules.atl_semantic_hub_v6 import ATLSemanticHubV6

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


def create_independent(nc, inst, feat, noise=0.5):
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


def disc_v6(atl, vis, lang, n=30):
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
    with torch.no_grad():
        acts = layer(F.normalize(features, dim=1), k=k)
        ucs = concepts.unique()
        within, between = [], []
        for c in ucs:
            mask_c = concepts == c
            if mask_c.sum() < 2:
                continue
            acts_c = acts[mask_c]
            n = len(acts_c)
            sim_c = acts_c @ acts_c.T
            mask_offdiag = ~torch.eye(n, dtype=torch.bool, device=features.device)
            within.append(sim_c[mask_offdiag].mean().item())
            acts_other = acts[~mask_c]
            if len(acts_other) > 0:
                idx = torch.randperm(len(acts_other))[:min(20, len(acts_other))]
                between.append((acts_c @ acts_other[idx].T).mean().item())
        w = np.mean(within) if within else 0
        b = np.mean(between) if between else 0
        return w - b


def train_cluster_only(atl, tv, tl, epochs):
    """Phase 1: Learn competitive prototypes only, no binding."""
    for _ in range(epochs):
        p = torch.randperm(len(tv), device=DEVICE)
        atl.vis_layer.learn_batch(F.normalize(tv[p], dim=1), k=atl.k)
        atl.lang_layer.learn_batch(F.normalize(tl[p], dim=1), k=atl.k)


def train_bind_only(atl, tv, tl, epochs):
    """Phase 2: Learn binding matrix only, competitive layers frozen."""
    # Save and disable prototype learning
    orig_vis_lr = atl.vis_layer.proto_lr.clone()
    orig_lang_lr = atl.lang_layer.proto_lr.clone()
    atl.vis_layer.proto_lr.zero_()
    atl.lang_layer.proto_lr.zero_()
    
    for _ in range(epochs):
        p = torch.randperm(len(tv), device=DEVICE)
        atl.bind_batch(tv[p], tl[p])
    
    # Restore
    atl.vis_layer.proto_lr.copy_(orig_vis_lr)
    atl.lang_layer.proto_lr.copy_(orig_lang_lr)


def train_joint(atl, tv, tl, epochs):
    """Joint training (original approach)."""
    for _ in range(epochs):
        p = torch.randperm(len(tv), device=DEVICE)
        atl.bind_batch(tv[p], tl[p])


# ============================================================
def exp1_two_phase():
    """Compare joint vs two-phase training at different noise levels."""
    print("\n" + "=" * 70)
    print("EXP 1: Joint vs Two-Phase Training")
    print("=" * 70)
    
    nc, inst, feat = 16, 20, 64
    
    print(f"\n{'Noise':<7} {'Joint Te':<10} {'2Phase Te':<10} {'J VGap':<10} {'2P VGap':<10}")
    print("-" * 50)
    
    for noise in [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        tv, tl, xv, xl, tc, xc = create_independent(nc, inst, feat, noise)
        n_protos = nc * 3
        
        # Joint (baseline)
        atl_j = ATLSemanticHubV6(
            feature_dim=feat, n_vis_protos=n_protos, n_lang_protos=n_protos,
            lr_bind=0.1, lr_unbind=0.02, lr_proto=0.05, k=3,
        ).to(DEVICE)
        train_joint(atl_j, tv, tl, 3000)
        te_j = disc_v6(atl_j, tv, tl), disc_v6(atl_j, xv, xl)
        cq_j = cluster_quality(atl_j.vis_layer, tv, tc, k=3)
        
        # Two-phase: 2000 cluster + 1000 bind
        atl_2 = ATLSemanticHubV6(
            feature_dim=feat, n_vis_protos=n_protos, n_lang_protos=n_protos,
            lr_bind=0.1, lr_unbind=0.02, lr_proto=0.05, k=3,
        ).to(DEVICE)
        train_cluster_only(atl_2, tv, tl, 2000)
        train_bind_only(atl_2, tv, tl, 1000)
        te_2 = disc_v6(atl_2, tv, tl), disc_v6(atl_2, xv, xl)
        cq_2 = cluster_quality(atl_2.vis_layer, tv, tc, k=3)
        
        print(f"{noise:<7.2f} {te_j[1]:<10.4f} {te_2[1]:<10.4f} {cq_j:<10.4f} {cq_2:<10.4f}")


# ============================================================
def exp2_more_data():
    """More training data per concept should improve clustering."""
    print("\n" + "=" * 70)
    print("EXP 2: More Data per Concept (Two-Phase)")
    print("=" * 70)
    
    nc, feat = 16, 64
    
    print(f"\n{'Inst':<7} {'Noise':<7} {'Train':<10} {'Test':<10} {'V Gap':<10} {'L Gap'}")
    print("-" * 52)
    
    for inst in [20, 40, 80, 160]:
        for noise in [0.2, 0.3, 0.4, 0.5]:
            tv, tl, xv, xl, tc, xc = create_independent(nc, inst, feat, noise)
            n_protos = nc * 3
            
            atl = ATLSemanticHubV6(
                feature_dim=feat, n_vis_protos=n_protos, n_lang_protos=n_protos,
                lr_bind=0.1, lr_unbind=0.02, lr_proto=0.05, k=3,
            ).to(DEVICE)
            train_cluster_only(atl, tv, tl, 2000)
            train_bind_only(atl, tv, tl, 1000)
            tr = disc_v6(atl, tv, tl)
            te = disc_v6(atl, xv, xl)
            cq_v = cluster_quality(atl.vis_layer, tv, tc, k=3)
            cq_l = cluster_quality(atl.lang_layer, tl, tc, k=3)
            print(f"{inst:<7} {noise:<7.1f} {tr:<10.4f} {te:<10.4f} {cq_v:<10.4f} {cq_l:.4f}")


# ============================================================
def exp3_best_config():
    """Find the best V6 configuration at noise=0.3 and 0.4."""
    print("\n" + "=" * 70)
    print("EXP 3: Configuration Search (noise=0.3, 0.4)")
    print("=" * 70)
    
    nc, feat = 16, 64
    
    configs = [
        # inst, k, n_protos, cluster_ep, bind_ep, lr_bind, lr_unbind
        (20,  2, 32, 3000, 2000, 0.15, 0.03),
        (20,  3, 48, 3000, 2000, 0.1,  0.02),
        (40,  2, 32, 3000, 2000, 0.15, 0.03),
        (40,  3, 48, 3000, 2000, 0.1,  0.02),
        (80,  2, 32, 5000, 3000, 0.15, 0.03),
        (80,  3, 48, 5000, 3000, 0.1,  0.02),
        (80,  2, 24, 5000, 3000, 0.2,  0.04),
        (80,  1, 20, 5000, 3000, 0.2,  0.04),
    ]
    
    for noise in [0.3, 0.4]:
        print(f"\n--- noise={noise} ---")
        print(f"{'Config':<30} {'Train':<10} {'Test':<10} {'V Gap':<10} {'Bind Sp'}")
        print("-" * 65)
        
        best_te, best_cfg = -1, ""
        for inst, k, np_, cep, bep, lr_b, lr_u in configs:
            tv, tl, xv, xl, tc, xc = create_independent(nc, inst, feat, noise)
            atl = ATLSemanticHubV6(
                feature_dim=feat, n_vis_protos=np_, n_lang_protos=np_,
                lr_bind=lr_b, lr_unbind=lr_u, lr_proto=0.05, k=k,
            ).to(DEVICE)
            train_cluster_only(atl, tv, tl, cep)
            train_bind_only(atl, tv, tl, bep)
            tr = disc_v6(atl, tv, tl)
            te = disc_v6(atl, xv, xl)
            cq = cluster_quality(atl.vis_layer, tv, tc, k=k)
            sp = atl.get_stats()['bind_sparsity']
            tag = f"i={inst},k={k},p={np_}"
            print(f"{tag:<30} {tr:<10.4f} {te:<10.4f} {cq:<10.4f} {sp:.4f}")
            if te > best_te:
                best_te, best_cfg = te, tag
        
        print(f"Best: {best_cfg} → test={best_te:.4f}")


# ============================================================
def exp4_scale_128():
    """Test best config at 128-dim (the previous failure point)."""
    print("\n" + "=" * 70)
    print("EXP 4: Scale to 128-dim (Two-Phase)")
    print("=" * 70)
    
    configs = [
        # feat, nc, inst, noise, k, np_, cep, bep
        (128, 16, 40, 0.2, 3, 48, 3000, 2000),
        (128, 16, 40, 0.3, 2, 32, 5000, 3000),
        (128, 16, 80, 0.3, 2, 32, 5000, 3000),
        (128, 16, 80, 0.4, 2, 32, 5000, 3000),
        (128, 50, 40, 0.2, 3, 150, 5000, 3000),
        (128, 50, 40, 0.3, 2, 100, 5000, 3000),
        (256, 16, 80, 0.3, 2, 32, 5000, 3000),
    ]
    
    print(f"\n{'Config':<35} {'Train':<10} {'Test':<10} {'V Gap':<10} {'Status'}")
    print("-" * 70)
    
    for feat, nc, inst, noise, k, np_, cep, bep in configs:
        tv, tl, xv, xl, tc, xc = create_independent(nc, inst, feat, noise)
        atl = ATLSemanticHubV6(
            feature_dim=feat, n_vis_protos=np_, n_lang_protos=np_,
            lr_bind=0.15, lr_unbind=0.03, lr_proto=0.05, k=k,
        ).to(DEVICE)
        train_cluster_only(atl, tv, tl, cep)
        train_bind_only(atl, tv, tl, bep)
        tr = disc_v6(atl, tv, tl)
        te = disc_v6(atl, xv, xl)
        cq = cluster_quality(atl.vis_layer, tv, tc, k=k)
        tag = f"{feat}d,{nc}c,{inst}i,n={noise},k={k},p={np_}"
        status = "✓" if te >= 0.15 else ("~" if te >= 0.05 else "✗")
        print(f"{tag:<35} {tr:<10.4f} {te:<10.4f} {cq:<10.4f} {status}")


def main():
    print("=" * 70)
    print("V6 IMPROVED: PUSHING NOISE TOLERANCE")
    print("=" * 70)
    exp1_two_phase()
    exp2_more_data()
    exp3_best_config()
    exp4_scale_128()
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
