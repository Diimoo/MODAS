#!/usr/bin/env python3
"""
Test ATL V3 - Biologically Plausible Hebbian Binding

Verify that ATL V3 achieves discrimination WITHOUT backprop:
- No optimizer.step()
- No .backward()
- Only local Hebbian updates
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modas.modules.atl_semantic_hub_v3 import ATLSemanticHubV3


def compute_discrimination(atl, test_pairs):
    """Compute discrimination using projected space similarity."""
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
    
    return np.mean(matched_sims) - np.mean(mismatched_sims), np.mean(matched_sims), np.mean(mismatched_sims)


def verify_no_backprop():
    """Verify that ATL V3 has no gradient-based parameters."""
    atl = ATLSemanticHubV3(n_prototypes=50, feature_dim=64, shared_dim=32)
    
    # Check that projection weights are buffers, not parameters
    n_params = sum(p.numel() for p in atl.parameters())
    print(f"Number of trainable parameters: {n_params}")
    
    if n_params == 0:
        print("✓ ATL V3 has NO trainable parameters (pure Hebbian)")
    else:
        print("✗ WARNING: ATL V3 has trainable parameters!")
        for name, param in atl.named_parameters():
            print(f"  - {name}: {param.shape}")
    
    return n_params == 0


def experiment1_correlated_synthetic():
    """Test on correlated synthetic data."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Correlated Synthetic Data (ATL V3 - Hebbian)")
    print("=" * 70)
    
    atl = ATLSemanticHubV3(
        n_prototypes=100,
        feature_dim=64,
        shared_dim=32,
        temperature=0.2,
        lr_proj=0.02,
        lr_proto=0.1,
    )
    
    # Create correlated data from shared concepts
    concept_embs = {}
    for concept in ['red', 'blue', 'green', 'yellow', 'circle', 'square', 'triangle', 'star']:
        concept_embs[concept] = F.normalize(torch.randn(64), dim=0)
    
    colors = ['red', 'blue', 'green', 'yellow']
    shapes = ['circle', 'square', 'triangle', 'star']
    
    train_data = []
    for _ in range(200):
        color = np.random.choice(colors)
        shape = np.random.choice(shapes)
        shared_base = concept_embs[color] + concept_embs[shape]
        vis = F.normalize(shared_base + 0.3 * torch.randn(64), dim=0)
        lang = F.normalize(shared_base + 0.3 * torch.randn(64), dim=0)
        train_data.append((vis, lang, f"{color} {shape}"))
    
    test_data = train_data[:20]
    
    # Input correlation (ceiling)
    input_sims = [F.cosine_similarity(v.unsqueeze(0), l.unsqueeze(0)).item() for v, l, _ in test_data]
    input_disc = np.mean(input_sims) - np.mean([
        F.cosine_similarity(test_data[i][0].unsqueeze(0), test_data[j][1].unsqueeze(0)).item()
        for i in range(len(test_data)) for j in range(len(test_data)) if i != j
    ])
    print(f"\nInput discrimination (ceiling): {input_disc:.4f}")
    
    # Initial discrimination
    init_disc, _, _ = compute_discrimination(atl, test_data)
    print(f"ATL V3 discrimination BEFORE training: {init_disc:.4f}")
    
    # Train with PURE HEBBIAN (no backprop!)
    print(f"\nTraining for 1000 epochs (pure Hebbian, no backprop)...")
    for epoch in range(1000):
        np.random.shuffle(train_data)
        margins = []
        
        for vis, lang, _ in train_data:
            # bind() does all Hebbian updates internally
            pos_sim, margin = atl.bind(vis, lang)
            margins.append(margin.item())
        
        if (epoch + 1) % 200 == 0:
            disc, matched, mismatched = compute_discrimination(atl, test_data)
            stats = atl.get_stats()
            print(f"  Epoch {epoch+1}: margin={np.mean(margins):.4f}, disc={disc:.4f}, "
                  f"capacity={stats['effective_capacity']:.1%}")
    
    # Final discrimination
    final_disc, final_matched, final_mismatched = compute_discrimination(atl, test_data)
    print(f"\nATL V3 discrimination AFTER training:")
    print(f"  Matched: {final_matched:.4f}, Mismatched: {final_mismatched:.4f}")
    print(f"  Discrimination: {final_disc:.4f}")
    
    if final_disc > 0.15:
        print("\nVERDICT: PASS (>0.15) - Hebbian learning works!")
    elif final_disc > 0.05:
        print("\nVERDICT: PARTIAL - Some learning, but weak")
    else:
        print("\nVERDICT: FAIL - Hebbian learning insufficient")
    
    return final_disc, input_disc


def experiment2_uncorrelated():
    """Test if ATL V3 can create alignment from uncorrelated features."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Can ATL V3 CREATE alignment from uncorrelated features?")
    print("=" * 70)
    
    atl = ATLSemanticHubV3(
        n_prototypes=100,
        feature_dim=64,
        shared_dim=32,
        temperature=0.2,
        lr_proj=0.05,  # Higher LR for harder task
        lr_proto=0.1,
    )
    
    # Create UNCORRELATED paired data
    n_concepts = 16
    samples_per_concept = 20
    
    train_data = []
    for concept_id in range(n_concepts):
        for _ in range(samples_per_concept):
            vis = F.normalize(torch.randn(64), dim=0)
            lang = F.normalize(torch.randn(64), dim=0)
            train_data.append((vis, lang, concept_id))
    
    test_data = [(train_data[i * samples_per_concept][0], 
                  train_data[i * samples_per_concept][1], i) 
                 for i in range(n_concepts)]
    
    # Verify zero input correlation
    input_sims = [F.cosine_similarity(v.unsqueeze(0), l.unsqueeze(0)).item() for v, l, _ in test_data]
    print(f"\nInput correlation: mean={np.mean(input_sims):.4f} (should be ~0)")
    
    # Initial discrimination
    init_disc, _, _ = compute_discrimination(atl, test_data)
    print(f"ATL V3 discrimination BEFORE training: {init_disc:.4f}")
    
    # Train
    print(f"\nTraining for 2000 epochs (this is the hard test)...")
    for epoch in range(2000):
        np.random.shuffle(train_data)
        margins = []
        
        for vis, lang, _ in train_data:
            pos_sim, margin = atl.bind(vis, lang)
            margins.append(margin.item())
        
        if (epoch + 1) % 500 == 0:
            disc, matched, mismatched = compute_discrimination(atl, test_data)
            print(f"  Epoch {epoch+1}: margin={np.mean(margins):.4f}, disc={disc:.4f}")
    
    # Final discrimination
    final_disc, final_matched, final_mismatched = compute_discrimination(atl, test_data)
    print(f"\nATL V3 discrimination AFTER training: {final_disc:.4f}")
    
    if final_disc > 0.1:
        print("\nVERDICT: Hebbian CAN create alignment from uncorrelated features!")
    elif final_disc > 0.05:
        print("\nVERDICT: PARTIAL - Some alignment learned")
    else:
        print("\nVERDICT: Hebbian cannot create alignment (expected - this is very hard)")
    
    return final_disc


def main():
    print("=" * 70)
    print("ATL V3 BINDING MECHANISM INVESTIGATION")
    print("=" * 70)
    print("\nTesting BIOLOGICALLY PLAUSIBLE ATL with:")
    print("  - Hebbian cross-correlation for projections")
    print("  - Three-factor Hebbian for prototypes")
    print("  - NO backprop, NO optimizer")
    
    # First verify no backprop
    print("\n--- Verifying architecture ---")
    is_pure_hebbian = verify_no_backprop()
    
    disc1, input1 = experiment1_correlated_synthetic()
    disc2 = experiment2_uncorrelated()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nPure Hebbian (no backprop): {'YES' if is_pure_hebbian else 'NO'}")
    print(f"Experiment 1 (correlated): {disc1:.4f} (input ceiling: {input1:.4f})")
    print(f"Experiment 2 (uncorrelated): {disc2:.4f}")
    
    print("\n" + "-" * 70)
    if disc1 > 0.1:
        print("CONCLUSION: ATL V3 (Hebbian) achieves meaningful discrimination!")
        print("Bio-plausibility maintained while learning cross-modal binding.")
    else:
        print("CONCLUSION: Pure Hebbian insufficient. May need architectural changes.")
    print("-" * 70)


if __name__ == '__main__':
    main()
