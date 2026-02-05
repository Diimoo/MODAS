#!/usr/bin/env python3
"""
MODAS Binding Investigation

Tests whether ATL binding mechanism actually works or needs rethinking.

Experiments:
1. Extended training (500 epochs) on synthetic correlated data
2. Full pipeline test with V1 + Language encoder
3. Analysis: does ATL CREATE alignment or just preserve input correlation?
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modas.modules import V1SparseCoding, LanguageEncoder, ATLSemanticHub
from modas.data.multimodal_dataset import SyntheticMultimodalDataset


def compute_discrimination(atl, test_pairs, use_embeddings=True):
    """Compute discrimination score."""
    matched_sims = []
    mismatched_sims = []
    
    for vis, lang, _ in test_pairs:
        if use_embeddings:
            sim = atl.compute_cross_modal_similarity(vis, lang, method='embedding')
        else:
            sim = F.cosine_similarity(vis.unsqueeze(0), lang.unsqueeze(0))
        matched_sims.append(sim.item())
    
    for i, (vis, _, _) in enumerate(test_pairs):
        for j, (_, lang, _) in enumerate(test_pairs):
            if i != j:
                if use_embeddings:
                    sim = atl.compute_cross_modal_similarity(vis, lang, method='embedding')
                else:
                    sim = F.cosine_similarity(vis.unsqueeze(0), lang.unsqueeze(0))
                mismatched_sims.append(sim.item())
    
    return np.mean(matched_sims) - np.mean(mismatched_sims), np.mean(matched_sims), np.mean(mismatched_sims)


def experiment1_extended_training():
    """
    Experiment 1: Extended training on synthetic correlated data.
    
    Question: Does discrimination improve with more training?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Extended Training on Correlated Synthetic Data")
    print("=" * 70)
    
    atl = ATLSemanticHub(n_prototypes=100, feature_dim=64, memory_size=200)
    
    # Create correlated data (same as demo - known issue)
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
    
    # Measure input correlation (what ATL receives)
    input_disc, input_matched, input_mismatched = compute_discrimination(atl, test_data, use_embeddings=False)
    print(f"\nInput feature correlation (what ATL receives):")
    print(f"  Matched mean: {input_matched:.4f}")
    print(f"  Mismatched mean: {input_mismatched:.4f}")
    print(f"  Input discrimination: {input_disc:.4f}")
    
    # Initial ATL discrimination (before training)
    init_disc, init_matched, init_mismatched = compute_discrimination(atl, test_data, use_embeddings=True)
    print(f"\nATL discrimination BEFORE training (via embeddings):")
    print(f"  Matched: {init_matched:.4f}, Mismatched: {init_mismatched:.4f}")
    print(f"  Discrimination: {init_disc:.4f}")
    
    # Train for 500 epochs
    print(f"\nTraining for 500 epochs...")
    for epoch in range(500):
        margins = []
        for vis, lang, _ in train_data:
            _, margin = atl.bind(vis, lang)
            margins.append(margin.item())
        
        if (epoch + 1) % 100 == 0:
            disc, matched, mismatched = compute_discrimination(atl, test_data, use_embeddings=True)
            print(f"  Epoch {epoch+1}: margin={np.mean(margins):.4f}, disc={disc:.4f} (matched={matched:.4f}, mismatched={mismatched:.4f})")
    
    # Final discrimination
    final_disc, final_matched, final_mismatched = compute_discrimination(atl, test_data, use_embeddings=True)
    print(f"\nATL discrimination AFTER training:")
    print(f"  Matched: {final_matched:.4f}, Mismatched: {final_mismatched:.4f}")
    print(f"  Discrimination: {final_disc:.4f}")
    
    improvement = final_disc - init_disc
    print(f"\nImprovement: {improvement:.4f}")
    print(f"Input discrimination (ceiling): {input_disc:.4f}")
    print(f"ATL discrimination / Input discrimination: {final_disc / input_disc:.2%}" if input_disc > 0 else "N/A")
    
    if final_disc > 0.15:
        print("\nVERDICT: PASS (>0.15)")
    elif final_disc > 0.1:
        print("\nVERDICT: MARGINAL (0.1-0.15)")
    else:
        print("\nVERDICT: FAIL (<0.1)")
    
    return final_disc, input_disc


def experiment2_full_pipeline():
    """
    Experiment 2: Full pipeline with V1 + Language encoder.
    
    Question: Does ATL work when V1 and Language encoder are in the loop?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Full Pipeline (V1 + Language Encoder)")
    print("=" * 70)
    
    # Create modules
    v1 = V1SparseCoding(n_bases=64, patch_size=8, n_channels=3)
    lang = LanguageEncoder(output_dim=64, load_pretrained=False)
    atl = ATLSemanticHub(n_prototypes=100, feature_dim=64, memory_size=200)
    
    # Create synthetic dataset with actual images
    dataset = SyntheticMultimodalDataset(n_samples=200, image_size=32, compositional=False)
    
    # Get features through full pipeline
    train_data = []
    print("\nExtracting features through V1 and Language encoder...")
    for i in range(len(dataset)):
        sample = dataset[i]
        vis_code = v1.forward(sample.image)
        lang_emb = lang.forward(sample.text)
        train_data.append((vis_code, lang_emb, sample.text))
    
    test_data = train_data[:20]
    
    # Measure input correlation (V1 output vs Language output)
    input_disc, input_matched, input_mismatched = compute_discrimination(atl, test_data, use_embeddings=False)
    print(f"\nInput feature correlation (V1 vs Language):")
    print(f"  Matched mean: {input_matched:.4f}")
    print(f"  Mismatched mean: {input_mismatched:.4f}")
    print(f"  Input discrimination: {input_disc:.4f}")
    
    if abs(input_disc) < 0.05:
        print("\n  WARNING: V1 and Language features have near-zero correlation!")
        print("  This is expected - they're in different representation spaces.")
        print("  ATL must CREATE alignment, not just preserve it.")
    
    # Initial ATL discrimination
    init_disc, init_matched, init_mismatched = compute_discrimination(atl, test_data, use_embeddings=True)
    print(f"\nATL discrimination BEFORE training:")
    print(f"  Discrimination: {init_disc:.4f}")
    
    # Train for 500 epochs
    print(f"\nTraining for 500 epochs...")
    for epoch in range(500):
        margins = []
        for vis, lang, _ in train_data:
            _, margin = atl.bind(vis, lang)
            margins.append(margin.item())
        
        if (epoch + 1) % 100 == 0:
            disc, matched, mismatched = compute_discrimination(atl, test_data, use_embeddings=True)
            print(f"  Epoch {epoch+1}: margin={np.mean(margins):.4f}, disc={disc:.4f}")
    
    # Final discrimination
    final_disc, final_matched, final_mismatched = compute_discrimination(atl, test_data, use_embeddings=True)
    print(f"\nATL discrimination AFTER training:")
    print(f"  Matched: {final_matched:.4f}, Mismatched: {final_mismatched:.4f}")
    print(f"  Discrimination: {final_disc:.4f}")
    
    if final_disc > 0.15:
        print("\nVERDICT: PASS - ATL creates alignment!")
    elif final_disc > 0.1:
        print("\nVERDICT: MARGINAL")
    else:
        print("\nVERDICT: FAIL - ATL cannot create alignment from uncorrelated features")
    
    return final_disc, input_disc


def experiment3_alignment_analysis():
    """
    Experiment 3: Does ATL CREATE alignment or just preserve it?
    
    Test: Feed UNCORRELATED features and see if ATL can learn to align them.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Can ATL CREATE alignment from uncorrelated features?")
    print("=" * 70)
    
    atl = ATLSemanticHub(n_prototypes=100, feature_dim=64, memory_size=200)
    
    # Create UNCORRELATED paired data
    # Same "concept" but features are independently random
    train_data = []
    concept_labels = []
    
    for i in range(200):
        label = i % 16  # 16 distinct concepts
        vis = F.normalize(torch.randn(64), dim=0)
        lang = F.normalize(torch.randn(64), dim=0)
        train_data.append((vis, lang, label))
        concept_labels.append(label)
    
    test_data = train_data[:32]  # 2 samples per concept
    
    # Verify zero correlation
    input_disc, input_matched, input_mismatched = compute_discrimination(atl, test_data, use_embeddings=False)
    print(f"\nInput correlation (should be ~0):")
    print(f"  Discrimination: {input_disc:.4f}")
    
    # Initial ATL discrimination
    init_disc, _, _ = compute_discrimination(atl, test_data, use_embeddings=True)
    print(f"\nATL discrimination BEFORE training: {init_disc:.4f}")
    
    # Train - can ATL learn that pair (i, i) should be more similar than (i, j)?
    print(f"\nTraining for 1000 epochs...")
    for epoch in range(1000):
        for vis, lang, _ in train_data:
            atl.bind(vis, lang)
        
        if (epoch + 1) % 200 == 0:
            disc, _, _ = compute_discrimination(atl, test_data, use_embeddings=True)
            print(f"  Epoch {epoch+1}: disc={disc:.4f}")
    
    # Final discrimination
    final_disc, _, _ = compute_discrimination(atl, test_data, use_embeddings=True)
    print(f"\nATL discrimination AFTER training: {final_disc:.4f}")
    
    if final_disc > 0.1:
        print("\nVERDICT: ATL CAN create alignment from uncorrelated features!")
    else:
        print("\nVERDICT: ATL CANNOT create alignment - it only preserves input correlation")
        print("         The binding mechanism needs rethinking.")
    
    return final_disc


def main():
    print("=" * 70)
    print("MODAS BINDING MECHANISM INVESTIGATION")
    print("=" * 70)
    print("\nThis investigation will determine whether ATL binding works in principle")
    print("or whether the architecture needs fundamental changes.")
    
    # Run experiments
    disc1, input1 = experiment1_extended_training()
    disc2, input2 = experiment2_full_pipeline()
    disc3 = experiment3_alignment_analysis()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nExperiment 1 (correlated synthetic): {disc1:.4f} (input ceiling: {input1:.4f})")
    print(f"Experiment 2 (full V1+Lang pipeline): {disc2:.4f} (input ceiling: {input2:.4f})")
    print(f"Experiment 3 (uncorrelated - can ATL create alignment?): {disc3:.4f}")
    
    print("\n" + "-" * 70)
    if disc1 > 0.15 and disc2 > 0.1:
        print("CONCLUSION: Binding mechanism works. Scale up training.")
    elif disc1 > 0.1 and disc3 < 0.05:
        print("CONCLUSION: ATL preserves but doesn't create alignment.")
        print("           Need architectural changes for cross-modal binding.")
    else:
        print("CONCLUSION: Binding mechanism fundamentally broken.")
        print("           Recommend investigating:")
        print("           1. Prototype initialization (k-means on features)")
        print("           2. Separate projection heads per modality")
        print("           3. Contrastive loss in embedding space, not feature space")
    print("-" * 70)


if __name__ == '__main__':
    main()
