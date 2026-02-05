#!/usr/bin/env python3
"""
Test ATL V2 redesigned binding mechanism.

Key changes being tested:
1. Sparse top-k activation (not sigmoid)
2. Modality projection heads (learnable)
3. InfoNCE-style contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modas.modules import V1SparseCoding, LanguageEncoder
from modas.modules.atl_semantic_hub_v2 import ATLSemanticHubV2
from modas.data.multimodal_dataset import SyntheticMultimodalDataset


def compute_discrimination(atl, test_pairs, modality1='visual', modality2='language'):
    """Compute discrimination using projected space similarity."""
    matched_sims = []
    mismatched_sims = []
    
    for vis, lang, _ in test_pairs:
        sim = atl.compute_cross_modal_similarity(vis, lang, modality1, modality2)
        matched_sims.append(sim.item())
    
    for i, (vis, _, _) in enumerate(test_pairs):
        for j, (_, lang, _) in enumerate(test_pairs):
            if i != j:
                sim = atl.compute_cross_modal_similarity(vis, lang, modality1, modality2)
                mismatched_sims.append(sim.item())
    
    return np.mean(matched_sims) - np.mean(mismatched_sims), np.mean(matched_sims), np.mean(mismatched_sims)


def experiment1_correlated_synthetic():
    """Test on correlated synthetic data."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Correlated Synthetic Data (ATL V2)")
    print("=" * 70)
    
    atl = ATLSemanticHubV2(n_prototypes=100, feature_dim=64, shared_dim=32, topk=5)
    atl.train()
    
    # Enable gradients for projection heads
    for param in atl.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam(atl.parameters(), lr=0.01)
    
    # Create correlated data
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
    
    # Input correlation
    input_sims_matched = []
    input_sims_mismatched = []
    for vis, lang, _ in test_data:
        input_sims_matched.append(F.cosine_similarity(vis.unsqueeze(0), lang.unsqueeze(0)).item())
    for i, (vis, _, _) in enumerate(test_data):
        for j, (_, lang, _) in enumerate(test_data):
            if i != j:
                input_sims_mismatched.append(F.cosine_similarity(vis.unsqueeze(0), lang.unsqueeze(0)).item())
    input_disc = np.mean(input_sims_matched) - np.mean(input_sims_mismatched)
    print(f"\nInput discrimination (ceiling): {input_disc:.4f}")
    
    # Initial discrimination
    init_disc, _, _ = compute_discrimination(atl, test_data)
    print(f"ATL V2 discrimination BEFORE training: {init_disc:.4f}")
    
    # Train with gradient descent
    print(f"\nTraining for 500 epochs...")
    for epoch in range(500):
        total_loss = 0
        margins = []
        
        optimizer.zero_grad()
        epoch_loss = torch.tensor(0.0)
        
        for vis, lang, _ in train_data:
            vis = vis.clone().requires_grad_(True)
            lang = lang.clone().requires_grad_(True)
            
            # Project to shared space
            vis_proj = atl.project(vis, 'visual')
            lang_proj = atl.project(lang, 'language')
            
            # Positive similarity
            pos_sim = F.cosine_similarity(vis_proj.unsqueeze(0), lang_proj.unsqueeze(0)).squeeze()
            
            # Get negatives from memory
            if len(atl.memory_lang) > 5:
                neg_indices = np.random.choice(len(atl.memory_lang), min(10, len(atl.memory_lang)), replace=False)
                neg_projs = torch.stack([atl.memory_lang[i] for i in neg_indices])
                neg_sims = (vis_proj.unsqueeze(0) @ neg_projs.T).squeeze()
                
                # InfoNCE loss
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims]) / 0.1
                labels = torch.zeros(1, dtype=torch.long)
                loss = F.cross_entropy(logits.unsqueeze(0), labels)
                margin = pos_sim - neg_sims.max()
            else:
                loss = -pos_sim
                margin = pos_sim
            
            epoch_loss = epoch_loss + loss
            margins.append(margin.item())
            
            # Store in memory
            atl.memory_vis.append(vis_proj.detach().clone())
            atl.memory_lang.append(lang_proj.detach().clone())
        
        # Backward pass
        epoch_loss = epoch_loss / len(train_data)
        epoch_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            disc, matched, mismatched = compute_discrimination(atl, test_data)
            print(f"  Epoch {epoch+1}: loss={epoch_loss.item():.4f}, margin={np.mean(margins):.4f}, disc={disc:.4f}")
    
    # Final discrimination
    final_disc, final_matched, final_mismatched = compute_discrimination(atl, test_data)
    print(f"\nATL V2 discrimination AFTER training:")
    print(f"  Matched: {final_matched:.4f}, Mismatched: {final_mismatched:.4f}")
    print(f"  Discrimination: {final_disc:.4f}")
    print(f"  Ratio to input ceiling: {final_disc / input_disc:.1%}" if input_disc > 0 else "")
    
    if final_disc > 0.15:
        print("\nVERDICT: PASS (>0.15)")
    elif final_disc > 0.1:
        print("\nVERDICT: MARGINAL (0.1-0.15)")
    else:
        print("\nVERDICT: FAIL (<0.1)")
    
    return final_disc, input_disc


def experiment2_full_pipeline():
    """Test with V1 + Language encoder in the loop."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Full Pipeline V1 + Language (ATL V2)")
    print("=" * 70)
    
    v1 = V1SparseCoding(n_bases=64, patch_size=8, n_channels=3)
    lang = LanguageEncoder(output_dim=64, load_pretrained=False)
    atl = ATLSemanticHubV2(n_prototypes=100, feature_dim=64, shared_dim=32, topk=5)
    atl.train()
    
    optimizer = torch.optim.Adam(atl.parameters(), lr=0.01)
    
    # Create dataset
    dataset = SyntheticMultimodalDataset(n_samples=200, image_size=32, compositional=False)
    
    # Extract features
    train_data = []
    print("\nExtracting features...")
    for i in range(len(dataset)):
        sample = dataset[i]
        vis_code = v1.forward(sample.image).detach()
        lang_emb = lang.forward(sample.text).detach()
        train_data.append((vis_code, lang_emb, sample.text))
    
    test_data = train_data[:20]
    
    # Input correlation
    input_sims = []
    for vis, lang_emb, _ in test_data:
        input_sims.append(F.cosine_similarity(vis.unsqueeze(0), lang_emb.unsqueeze(0)).item())
    print(f"\nInput correlation (V1 vs Lang): mean={np.mean(input_sims):.4f}, std={np.std(input_sims):.4f}")
    
    # Initial discrimination
    init_disc, _, _ = compute_discrimination(atl, test_data)
    print(f"ATL V2 discrimination BEFORE training: {init_disc:.4f}")
    
    # Train
    print(f"\nTraining for 500 epochs...")
    for epoch in range(500):
        optimizer.zero_grad()
        epoch_loss = torch.tensor(0.0)
        margins = []
        
        for vis, lang_emb, _ in train_data:
            vis = vis.clone().requires_grad_(True)
            lang_emb = lang_emb.clone().requires_grad_(True)
            
            vis_proj = atl.project(vis, 'visual')
            lang_proj = atl.project(lang_emb, 'language')
            
            pos_sim = F.cosine_similarity(vis_proj.unsqueeze(0), lang_proj.unsqueeze(0)).squeeze()
            
            if len(atl.memory_lang) > 5:
                neg_indices = np.random.choice(len(atl.memory_lang), min(10, len(atl.memory_lang)), replace=False)
                neg_projs = torch.stack([atl.memory_lang[i] for i in neg_indices])
                neg_sims = (vis_proj.unsqueeze(0) @ neg_projs.T).squeeze()
                
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims]) / 0.1
                labels = torch.zeros(1, dtype=torch.long)
                loss = F.cross_entropy(logits.unsqueeze(0), labels)
                margin = pos_sim - neg_sims.max()
            else:
                loss = -pos_sim
                margin = pos_sim
            
            epoch_loss = epoch_loss + loss
            margins.append(margin.item())
            
            atl.memory_vis.append(vis_proj.detach().clone())
            atl.memory_lang.append(lang_proj.detach().clone())
        
        epoch_loss = epoch_loss / len(train_data)
        epoch_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            disc, matched, mismatched = compute_discrimination(atl, test_data)
            print(f"  Epoch {epoch+1}: loss={epoch_loss.item():.4f}, margin={np.mean(margins):.4f}, disc={disc:.4f}")
    
    # Final discrimination
    final_disc, final_matched, final_mismatched = compute_discrimination(atl, test_data)
    print(f"\nATL V2 discrimination AFTER training:")
    print(f"  Matched: {final_matched:.4f}, Mismatched: {final_mismatched:.4f}")
    print(f"  Discrimination: {final_disc:.4f}")
    
    if final_disc > 0.15:
        print("\nVERDICT: PASS - ATL V2 creates alignment!")
    elif final_disc > 0.1:
        print("\nVERDICT: MARGINAL")
    else:
        print("\nVERDICT: FAIL")
    
    return final_disc


def experiment3_uncorrelated():
    """Test if ATL V2 can create alignment from uncorrelated features."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Can ATL V2 CREATE alignment from uncorrelated features?")
    print("=" * 70)
    
    atl = ATLSemanticHubV2(n_prototypes=100, feature_dim=64, shared_dim=32, topk=5)
    atl.train()
    
    optimizer = torch.optim.Adam(atl.parameters(), lr=0.01)
    
    # Create UNCORRELATED paired data with consistent concept labels
    train_data = []
    n_concepts = 16
    samples_per_concept = 20
    
    # For each concept, create multiple random vis-lang pairs
    concept_vis = {i: [] for i in range(n_concepts)}
    concept_lang = {i: [] for i in range(n_concepts)}
    
    for concept_id in range(n_concepts):
        for _ in range(samples_per_concept):
            vis = F.normalize(torch.randn(64), dim=0)
            lang = F.normalize(torch.randn(64), dim=0)
            concept_vis[concept_id].append(vis)
            concept_lang[concept_id].append(lang)
            train_data.append((vis, lang, concept_id))
    
    # Test data: one sample per concept
    test_data = [(concept_vis[i][0], concept_lang[i][0], i) for i in range(n_concepts)]
    
    # Verify zero input correlation
    input_sims = [F.cosine_similarity(v.unsqueeze(0), l.unsqueeze(0)).item() for v, l, _ in test_data]
    print(f"\nInput correlation: mean={np.mean(input_sims):.4f} (should be ~0)")
    
    # Initial discrimination
    init_disc, _, _ = compute_discrimination(atl, test_data)
    print(f"ATL V2 discrimination BEFORE training: {init_disc:.4f}")
    
    # Train - can ATL learn that pair (i, i) should be similar?
    print(f"\nTraining for 1000 epochs...")
    for epoch in range(1000):
        np.random.shuffle(train_data)
        optimizer.zero_grad()
        epoch_loss = torch.tensor(0.0)
        
        for vis, lang, concept_id in train_data:
            vis = vis.clone().requires_grad_(True)
            lang = lang.clone().requires_grad_(True)
            
            vis_proj = atl.project(vis, 'visual')
            lang_proj = atl.project(lang, 'language')
            
            pos_sim = F.cosine_similarity(vis_proj.unsqueeze(0), lang_proj.unsqueeze(0)).squeeze()
            
            if len(atl.memory_lang) > 5:
                neg_indices = np.random.choice(len(atl.memory_lang), min(10, len(atl.memory_lang)), replace=False)
                neg_projs = torch.stack([atl.memory_lang[i] for i in neg_indices])
                neg_sims = (vis_proj.unsqueeze(0) @ neg_projs.T).squeeze()
                
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims]) / 0.1
                labels = torch.zeros(1, dtype=torch.long)
                loss = F.cross_entropy(logits.unsqueeze(0), labels)
            else:
                loss = -pos_sim
            
            epoch_loss = epoch_loss + loss
            
            atl.memory_vis.append(vis_proj.detach().clone())
            atl.memory_lang.append(lang_proj.detach().clone())
        
        epoch_loss = epoch_loss / len(train_data)
        epoch_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            disc, matched, mismatched = compute_discrimination(atl, test_data)
            print(f"  Epoch {epoch+1}: loss={epoch_loss.item():.4f}, disc={disc:.4f}")
    
    # Final discrimination
    final_disc, final_matched, final_mismatched = compute_discrimination(atl, test_data)
    print(f"\nATL V2 discrimination AFTER training: {final_disc:.4f}")
    
    if final_disc > 0.1:
        print("\nVERDICT: ATL V2 CAN create alignment from uncorrelated features!")
    else:
        print("\nVERDICT: ATL V2 still cannot create alignment")
    
    return final_disc


def main():
    print("=" * 70)
    print("ATL V2 BINDING MECHANISM INVESTIGATION")
    print("=" * 70)
    print("\nTesting redesigned ATL with:")
    print("  - Sparse top-k activation")
    print("  - Learnable modality projection heads")
    print("  - InfoNCE-style contrastive loss")
    
    disc1, input1 = experiment1_correlated_synthetic()
    disc2 = experiment2_full_pipeline()
    disc3 = experiment3_uncorrelated()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nExperiment 1 (correlated synthetic): {disc1:.4f} (input ceiling: {input1:.4f})")
    print(f"Experiment 2 (full V1+Lang pipeline): {disc2:.4f}")
    print(f"Experiment 3 (uncorrelated - create alignment?): {disc3:.4f}")
    
    print("\n" + "-" * 70)
    if disc1 > 0.15 and disc2 > 0.1:
        print("CONCLUSION: ATL V2 WORKS. Projection heads enable cross-modal binding.")
    elif disc3 > 0.1:
        print("CONCLUSION: ATL V2 can create alignment! Architecture is sound.")
    else:
        print("CONCLUSION: ATL V2 still has issues. Further investigation needed.")
    print("-" * 70)


if __name__ == '__main__':
    main()
