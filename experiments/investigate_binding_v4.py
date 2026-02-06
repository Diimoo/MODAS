#!/usr/bin/env python3
"""
Test ATL V4 - Collapse Prevention and Generalization

Key tests:
1. Long-run collapse test (10,000 epochs) - does discrimination degrade?
2. Generalization test - train/test split by INSTANCE, not pair
3. Collapse metric tracking - is projection variance maintained?
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modas.modules.atl_semantic_hub_v3 import ATLSemanticHubV3
from modas.modules.atl_semantic_hub_v4 import ATLSemanticHubV4


def compute_discrimination(atl, test_pairs):
    """Compute discrimination on test pairs."""
    matched_sims = []
    mismatched_sims = []
    
    for vis, lang, _ in test_pairs:
        sim = atl.compute_cross_modal_similarity(vis, lang, 'visual', 'language')
        matched_sims.append(sim.item())
    
    for i, (vis, _, _) in enumerate(test_pairs):
        for j, (_, lang, _) in enumerate(test_pairs):
            if i != j:
                sim = atl.compute_cross_modal_similarity(vis, lang, 'visual', 'language')
                mismatched_sims.append(sim.item())
    
    return np.mean(matched_sims) - np.mean(mismatched_sims)


def compute_projection_variance(atl, data, modality='visual'):
    """Compute variance of projections - low = collapse."""
    projs = []
    for vis, lang, _ in data[:50]:
        feat = vis if modality == 'visual' else lang
        proj = atl.project(feat, modality)
        projs.append(proj.detach())
    
    projs = torch.stack(projs)
    return projs.var(dim=0).mean().item()


def experiment1_collapse_test():
    """Test long-run collapse: 10,000 epochs."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Long-Run Collapse Test (10,000 epochs)")
    print("=" * 70)
    
    # Create correlated data
    concept_embs = {c: F.normalize(torch.randn(64), dim=0) 
                    for c in ['red', 'blue', 'green', 'yellow', 'circle', 'square', 'triangle', 'star']}
    
    train_data = []
    for _ in range(200):
        color = np.random.choice(['red', 'blue', 'green', 'yellow'])
        shape = np.random.choice(['circle', 'square', 'triangle', 'star'])
        shared = concept_embs[color] + concept_embs[shape]
        vis = F.normalize(shared + 0.3 * torch.randn(64), dim=0)
        lang = F.normalize(shared + 0.3 * torch.randn(64), dim=0)
        train_data.append((vis, lang, f"{color} {shape}"))
    
    test_data = train_data[:20]
    
    # Test both V3 and V4
    for version, ATLClass in [('V3', ATLSemanticHubV3), ('V4', ATLSemanticHubV4)]:
        print(f"\n--- ATL {version} ---")
        
        if version == 'V3':
            atl = ATLClass(n_prototypes=100, feature_dim=64, shared_dim=32, temperature=0.2)
        else:
            atl = ATLClass(n_prototypes=100, feature_dim=64, shared_dim=32, temperature=0.2,
                          margin_threshold=-0.3)  # Allow updates when margin > -0.3
        
        disc_history = []
        var_history = []
        
        for epoch in range(10000):
            np.random.shuffle(train_data)
            for vis, lang, _ in train_data:
                atl.bind(vis, lang)
            
            if (epoch + 1) % 1000 == 0:
                disc = compute_discrimination(atl, test_data)
                vis_var = compute_projection_variance(atl, train_data, 'visual')
                lang_var = compute_projection_variance(atl, train_data, 'language')
                disc_history.append(disc)
                var_history.append((vis_var + lang_var) / 2)
                print(f"  Epoch {epoch+1}: disc={disc:.4f}, proj_var={var_history[-1]:.6f}")
        
        # Check for collapse
        if len(disc_history) >= 3:
            early_disc = disc_history[0]
            late_disc = disc_history[-1]
            degradation = early_disc - late_disc
            
            if degradation > 0.1:
                print(f"\n  ⚠ COLLAPSE DETECTED: disc dropped from {early_disc:.3f} to {late_disc:.3f}")
            elif late_disc > 0.15:
                print(f"\n  ✓ STABLE: disc maintained at {late_disc:.3f}")
            else:
                print(f"\n  ✗ FAILED: disc too low at {late_disc:.3f}")


def experiment2_generalization():
    """
    Test GENERALIZATION, not memorization.
    
    Key: train on 15 instances per concept, test on held-out 5 instances.
    The model should recognize that a NOVEL (vis, lang) pair from the same
    concept should be similar, even though it never saw that exact pair.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Generalization Test (train/test split by INSTANCE)")
    print("=" * 70)
    
    n_concepts = 16
    instances_per_concept = 20
    train_instances = 15
    test_instances = 5
    
    # Create concept embeddings
    concept_protos = [F.normalize(torch.randn(64), dim=0) for _ in range(n_concepts)]
    
    # Generate instances per concept
    all_data = {c: [] for c in range(n_concepts)}
    for c in range(n_concepts):
        for _ in range(instances_per_concept):
            # Each instance is a noisy version of concept prototype
            vis = F.normalize(concept_protos[c] + 0.5 * torch.randn(64), dim=0)
            lang = F.normalize(concept_protos[c] + 0.5 * torch.randn(64), dim=0)
            all_data[c].append((vis, lang, c))
    
    # Split train/test BY INSTANCE (not by pair)
    train_data = []
    test_data = []
    for c in range(n_concepts):
        np.random.shuffle(all_data[c])
        train_data.extend(all_data[c][:train_instances])
        test_data.extend(all_data[c][train_instances:])
    
    print(f"\nTrain: {len(train_data)} pairs ({train_instances} per concept)")
    print(f"Test: {len(test_data)} pairs ({test_instances} per concept) - NOVEL INSTANCES")
    
    for version, ATLClass in [('V3', ATLSemanticHubV3), ('V4', ATLSemanticHubV4)]:
        print(f"\n--- ATL {version} ---")
        
        if version == 'V3':
            atl = ATLClass(n_prototypes=100, feature_dim=64, shared_dim=32, temperature=0.2)
        else:
            atl = ATLClass(n_prototypes=100, feature_dim=64, shared_dim=32, temperature=0.2,
                          margin_threshold=-0.3)
        
        # Train
        print("Training for 2000 epochs...")
        for epoch in range(2000):
            np.random.shuffle(train_data)
            for vis, lang, _ in train_data:
                atl.bind(vis, lang)
            
            if (epoch + 1) % 500 == 0:
                train_disc = compute_discrimination(atl, train_data[:20])
                test_disc = compute_discrimination(atl, test_data[:20])
                print(f"  Epoch {epoch+1}: train_disc={train_disc:.4f}, test_disc={test_disc:.4f}")
        
        # Final evaluation
        train_disc = compute_discrimination(atl, train_data[:30])
        test_disc = compute_discrimination(atl, test_data)
        
        print(f"\nFinal results:")
        print(f"  Train discrimination: {train_disc:.4f}")
        print(f"  Test discrimination:  {test_disc:.4f} (NOVEL INSTANCES)")
        
        generalization_ratio = test_disc / train_disc if train_disc > 0 else 0
        print(f"  Generalization ratio: {generalization_ratio:.1%}")
        
        if test_disc > 0.1:
            print("\n  ✓ GENERALIZES: learns alignment, not memorization")
        elif test_disc > 0.05:
            print("\n  ~ PARTIAL: some generalization")
        else:
            print("\n  ✗ MEMORIZATION: only works on training pairs")


def experiment3_collapse_metric():
    """Track collapse metric over training."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Collapse Metric Tracking")
    print("=" * 70)
    
    concept_embs = {c: F.normalize(torch.randn(64), dim=0) 
                    for c in ['red', 'blue', 'green', 'yellow', 'circle', 'square', 'triangle', 'star']}
    
    train_data = []
    for _ in range(200):
        color = np.random.choice(['red', 'blue', 'green', 'yellow'])
        shape = np.random.choice(['circle', 'square', 'triangle', 'star'])
        shared = concept_embs[color] + concept_embs[shape]
        vis = F.normalize(shared + 0.3 * torch.randn(64), dim=0)
        lang = F.normalize(shared + 0.3 * torch.randn(64), dim=0)
        train_data.append((vis, lang, f"{color} {shape}"))
    
    print("\nTracking projection variance (low = collapse):\n")
    print(f"{'Epoch':<10} {'V3 Var':<12} {'V4 Var':<12} {'V3 Disc':<12} {'V4 Disc':<12}")
    print("-" * 58)
    
    atl_v3 = ATLSemanticHubV3(n_prototypes=100, feature_dim=64, shared_dim=32)
    atl_v4 = ATLSemanticHubV4(n_prototypes=100, feature_dim=64, shared_dim=32, margin_threshold=-0.3)
    
    for epoch in range(5000):
        np.random.shuffle(train_data)
        for vis, lang, _ in train_data:
            atl_v3.bind(vis, lang)
            atl_v4.bind(vis, lang)
        
        if (epoch + 1) % 500 == 0:
            v3_var = compute_projection_variance(atl_v3, train_data)
            v4_var = compute_projection_variance(atl_v4, train_data)
            v3_disc = compute_discrimination(atl_v3, train_data[:20])
            v4_disc = compute_discrimination(atl_v4, train_data[:20])
            print(f"{epoch+1:<10} {v3_var:<12.6f} {v4_var:<12.6f} {v3_disc:<12.4f} {v4_disc:<12.4f}")


def main():
    print("=" * 70)
    print("ATL V4 INVESTIGATION: Collapse Prevention & Generalization")
    print("=" * 70)
    
    experiment1_collapse_test()
    experiment2_generalization()
    experiment3_collapse_metric()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key questions answered:
1. Does V3 collapse over long training? (Check disc degradation)
2. Does V4's anti-Hebbian term prevent collapse? (Check variance)
3. Do models generalize to novel instances? (Test disc vs train disc)
""")


if __name__ == '__main__':
    main()
