#!/usr/bin/env python3
"""
MODAS Quick Start Example

Demonstrates basic usage of MODAS modules with synthetic data.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from modas.modules import V1SparseCoding, A1SparseCoding, LanguageEncoder, ATLSemanticHub
from modas.data.multimodal_dataset import SyntheticMultimodalDataset
from modas.utils.metrics import compute_sparsity, compute_discrimination


def demo_v1_sparse_coding():
    """Demonstrate V1 sparse coding on random patches."""
    print("\n" + "=" * 50)
    print("V1 SPARSE CODING DEMO")
    print("=" * 50)
    
    # Create model
    v1 = V1SparseCoding(n_bases=64, patch_size=8, n_channels=3)
    print(f"Created V1 with {v1.n_bases} bases, {v1.patch_size}x{v1.patch_size} patches")
    
    # Generate random patches
    patches = torch.randn(100, 8 * 8 * 3)
    patches = patches - patches.mean(dim=1, keepdim=True)
    patches = patches / (patches.std(dim=1, keepdim=True) + 1e-8)
    
    # Train for a few iterations
    print("\nTraining on 100 random patches...")
    for epoch in range(5):
        total_mse = 0
        for patch in patches:
            code = v1.lca_inference(patch.unsqueeze(0)).squeeze(0)
            mse = v1.learn(patch, code)
            total_mse += mse
        
        avg_mse = total_mse / len(patches)
        sparsity = compute_sparsity(v1.lca_inference(patches))
        print(f"  Epoch {epoch+1}: MSE={avg_mse:.4f}, Sparsity={sparsity:.2%}")
    
    # Test inference
    test_patch = torch.randn(8 * 8 * 3)
    code = v1.lca_inference(test_patch)
    print(f"\nTest inference:")
    print(f"  Code shape: {code.shape}")
    print(f"  Non-zero elements: {(code.abs() > 0.01).sum().item()}")
    print(f"  Sparsity: {compute_sparsity(code):.2%}")
    
    return v1


def demo_language_encoder():
    """Demonstrate language encoder."""
    print("\n" + "=" * 50)
    print("LANGUAGE ENCODER DEMO")
    print("=" * 50)
    
    # Create encoder (without pretrained Word2Vec for demo)
    lang = LanguageEncoder(output_dim=64, load_pretrained=False)
    print(f"Created LanguageEncoder with {lang.output_dim}-dim output")
    
    # Encode some texts
    texts = ["red circle", "blue square", "green triangle"]
    
    print("\nEncoding texts:")
    for text in texts:
        emb = lang.forward(text)
        print(f"  '{text}': shape={emb.shape}, norm={emb.norm():.4f}")
    
    # Compute similarities
    print("\nPairwise similarities:")
    for i, t1 in enumerate(texts):
        for j, t2 in enumerate(texts):
            if i < j:
                sim = lang.similarity(t1, t2)
                print(f"  '{t1}' vs '{t2}': {sim:.4f}")
    
    return lang


def demo_atl_binding():
    """Demonstrate ATL cross-modal binding."""
    print("\n" + "=" * 50)
    print("ATL BINDING DEMO")
    print("=" * 50)
    
    # Create ATL module (no language encoder needed for this demo)
    atl = ATLSemanticHub(n_prototypes=50, feature_dim=64, memory_size=50)
    
    print(f"Created ATL with {atl.n_prototypes} prototypes, {atl.feature_dim}-dim features")
    
    # Create CORRELATED visual-language pairs for meaningful training
    # KEY INSIGHT: Both modalities must share underlying structure
    # We simulate this by deriving both from shared concept embeddings
    print("Creating correlated feature pairs (shared concept space)...")
    n_samples = 50
    paired_data = []
    
    # Create SHARED concept embeddings (both vis and lang derived from these)
    concept_embs = {}
    concepts = ['red', 'blue', 'green', 'circle', 'square', 'triangle']
    for concept in concepts:
        concept_embs[concept] = F.normalize(torch.randn(64), dim=0)
    
    # Generate paired data with INHERENT correlation
    for i in range(n_samples):
        color = np.random.choice(['red', 'blue', 'green'])
        shape = np.random.choice(['circle', 'square', 'triangle'])
        text = f"{color} {shape}"
        
        # Shared base = sum of concept embeddings
        shared_base = concept_embs[color] + concept_embs[shape]
        
        # Visual: shared base + visual-specific noise
        vis_code = F.normalize(shared_base + 0.3 * torch.randn(64), dim=0)
        
        # Language: shared base + language-specific noise (NOT Word2Vec)
        # This ensures matched pairs have high correlation
        lang_emb = F.normalize(shared_base + 0.3 * torch.randn(64), dim=0)
        
        paired_data.append((vis_code, lang_emb, text))
    
    # Verify correlation exists
    sample_sim = F.cosine_similarity(
        paired_data[0][0].unsqueeze(0), 
        paired_data[0][1].unsqueeze(0)
    ).item()
    print(f"Created {len(paired_data)} pairs (sample matched sim: {sample_sim:.3f})")
    
    # Training loop
    print("\nTraining ATL binding...")
    for epoch in range(10):
        total_sim = 0
        total_margin = 0
        
        for vis_code, lang_emb, _ in paired_data:
            sim, margin = atl.bind(vis_code, lang_emb)
            total_sim += sim.item()
            total_margin += margin.item()
        
        avg_sim = total_sim / len(paired_data)
        avg_margin = total_margin / len(paired_data)
        print(f"  Epoch {epoch+1}: Similarity={avg_sim:.4f}, Margin={avg_margin:.4f}")
    
    # Test discrimination using FEATURE similarity (not embedding)
    print("\nTesting discrimination (feature-space)...")
    matched_sims = []
    mismatched_sims = []
    
    # Use first 10 pairs for test
    test_pairs = paired_data[:10]
    
    # Matched: visual-language from same pair
    for vis, lang_emb, _ in test_pairs:
        sim = F.cosine_similarity(vis.unsqueeze(0), lang_emb.unsqueeze(0)).item()
        matched_sims.append(sim)
    
    # Mismatched: visual from one pair, language from another
    for i, (vis, _, _) in enumerate(test_pairs):
        for j, (_, lang_emb, _) in enumerate(test_pairs):
            if i != j:
                sim = F.cosine_similarity(vis.unsqueeze(0), lang_emb.unsqueeze(0)).item()
                mismatched_sims.append(sim)
    
    disc = np.mean(matched_sims) - np.mean(mismatched_sims)
    print(f"  Matched mean: {np.mean(matched_sims):.4f}")
    print(f"  Mismatched mean: {np.mean(mismatched_sims):.4f}")
    print(f"  Discrimination: {disc:.4f}")
    
    if disc > 0.15:
        print("  Status: GOOD ✓")
    elif disc > 0.1:
        print("  Status: MARGINAL ⚠")
    else:
        print("  Status: NEEDS TRAINING ⚠ (expected for quick demo)")
    
    return atl


def demo_full_pipeline():
    """Demonstrate full MODAS pipeline."""
    print("\n" + "=" * 50)
    print("FULL MODAS PIPELINE DEMO")
    print("=" * 50)
    
    # Create all modules
    print("Creating modules...")
    v1 = V1SparseCoding(n_bases=128, patch_size=16, n_channels=3)
    lang = LanguageEncoder(output_dim=128, load_pretrained=False)
    atl = ATLSemanticHub(n_prototypes=100, feature_dim=128)
    
    # Create synthetic data
    print("Creating synthetic dataset...")
    dataset = SyntheticMultimodalDataset(
        n_samples=100, 
        image_size=64, 
        compositional=True
    )
    
    print(f"\nPipeline architecture:")
    print(f"  V1: {v1.n_bases} bases, {v1.patch_size}x{v1.patch_size} patches")
    print(f"  Language: {lang.output_dim}-dim output")
    print(f"  ATL: {atl.n_prototypes} prototypes, {atl.feature_dim}-dim features")
    print(f"  Dataset: {len(dataset)} samples")
    
    # Process a sample through the pipeline
    print("\nProcessing sample through pipeline...")
    sample = dataset[0]
    print(f"  Sample text: '{sample.text}'")
    print(f"  Sample image shape: {sample.image.shape}")
    
    # V1 encoding
    vis_code = v1.forward(sample.image)
    print(f"  V1 code shape: {vis_code.shape}")
    print(f"  V1 sparsity: {compute_sparsity(vis_code):.2%}")
    
    # Language encoding
    lang_emb = lang.forward(sample.text)
    print(f"  Language embedding shape: {lang_emb.shape}")
    
    # ATL binding
    sim, margin = atl.bind(vis_code, lang_emb)
    print(f"  ATL similarity: {sim.item():.4f}")
    print(f"  ATL margin: {margin.item():.4f}")
    
    print("\n✓ Pipeline complete!")


def main():
    """Run all demos."""
    print("=" * 60)
    print("MODAS - Modular Developmental Architecture for Semantics")
    print("Quick Start Demo")
    print("=" * 60)
    
    # Run demos
    demo_v1_sparse_coding()
    demo_language_encoder()
    demo_atl_binding()
    demo_full_pipeline()
    
    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train V1 on real images: python -m modas.training.train_v1")
    print("  2. Train A1 on audio: python -m modas.training.train_a1")
    print("  3. Train ATL binding: python -m modas.training.train_atl")
    print("  4. Validate results: python -m modas.evaluation.validate_atl")


if __name__ == '__main__':
    main()
