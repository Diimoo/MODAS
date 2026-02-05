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
    
    # Create modules
    v1 = V1SparseCoding(n_bases=64, patch_size=8, n_channels=3)
    lang = LanguageEncoder(output_dim=64, load_pretrained=False)
    atl = ATLSemanticHub(n_prototypes=50, feature_dim=64, memory_size=20)
    
    print(f"Created ATL with {atl.n_prototypes} prototypes, {atl.feature_dim}-dim features")
    
    # Create synthetic dataset
    dataset = SyntheticMultimodalDataset(n_samples=50, image_size=32, compositional=False)
    print(f"Created synthetic dataset with {len(dataset)} samples")
    
    # Training loop
    print("\nTraining ATL binding...")
    for epoch in range(5):
        total_sim = 0
        total_margin = 0
        
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # Get features
            # For demo, use random features (normally would use v1.forward(image))
            vis_code = F.normalize(torch.randn(64), dim=0)
            lang_emb = lang.forward(sample.text)
            
            # Bind
            sim, margin = atl.bind(vis_code, lang_emb)
            total_sim += sim.item()
            total_margin += margin.item()
        
        avg_sim = total_sim / len(dataset)
        avg_margin = total_margin / len(dataset)
        print(f"  Epoch {epoch+1}: Similarity={avg_sim:.4f}, Margin={avg_margin:.4f}")
    
    # Test discrimination
    print("\nTesting discrimination...")
    matched_sims = []
    mismatched_sims = []
    
    # Get features for test
    test_features = []
    for i in range(10):
        sample = dataset[i]
        vis = F.normalize(torch.randn(64), dim=0)
        lang = lang_encoder.forward(sample.text) if 'lang_encoder' in dir() else F.normalize(torch.randn(64), dim=0)
        test_features.append((vis, lang))
    
    # Matched
    for vis, lang in test_features:
        sim = atl.compute_cross_modal_similarity(vis, lang)
        matched_sims.append(sim.item())
    
    # Mismatched
    for i, (vis, _) in enumerate(test_features):
        for j, (_, lang) in enumerate(test_features):
            if i != j:
                sim = atl.compute_cross_modal_similarity(vis, lang)
                mismatched_sims.append(sim.item())
    
    disc = np.mean(matched_sims) - np.mean(mismatched_sims)
    print(f"  Matched mean: {np.mean(matched_sims):.4f}")
    print(f"  Mismatched mean: {np.mean(mismatched_sims):.4f}")
    print(f"  Discrimination: {disc:.4f}")
    
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
