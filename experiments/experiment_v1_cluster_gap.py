#!/usr/bin/env python3
"""
MODAS Experiment: V1 LCA Cluster Gap on Real Images (CIFAR-10)

THE CRITICAL QUESTION:
Does V1's LCA sparse coding produce codes where within-category similarity
exceeds between-category similarity by ≥ 0.20 (the V6 binding threshold)?

If yes → V6 associative binding is viable with real V1 output
If no  → V1 needs architectural changes or an intermediate layer is needed

WHAT THIS SCRIPT DOES:
1. Downloads CIFAR-10 (32×32 RGB images, 10 categories)
2. Trains V1 LCA on image patches (Hebbian dictionary learning)
3. Computes sparse codes for test images grouped by category
4. Measures within-category vs between-category cosine similarity
5. Reports cluster gap and compares to 0.20 threshold

USAGE:
    cd /home/ahmed/Dokumente/MODAS
    source venv/bin/activate
    python experiments/experiment_v1_cluster_gap.py --device cuda --n_epochs 50

    # Full sweep (slower, ~15 min on GPU, tests 8 configs)
    python experiments/experiment_v1_cluster_gap.py --device cuda --n_epochs 50 --sweep
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import sys
from pathlib import Path
from collections import defaultdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    print("ERROR: torchvision required. Install with: pip install torchvision")
    sys.exit(1)

from modas.modules.v1_sparse_coding import V1SparseCoding, extract_patches


# ============================================================================
# CONFIGURATION
# ============================================================================

# CIFAR-10 categories for reference
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# V6 binding threshold (from V6 experiments on synthetic data)
CLUSTER_GAP_THRESHOLD = 0.20


def parse_args():
    parser = argparse.ArgumentParser(description='V1 Cluster Gap Experiment')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_bases', type=int, default=128, help='Number of dictionary bases')
    parser.add_argument('--patch_size', type=int, default=8, help='Patch size (8 for CIFAR-10 32x32)')
    parser.add_argument('--lambda_sparse', type=float, default=0.5, help='Sparsity penalty')
    parser.add_argument('--n_epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--train_samples', type=int, default=10000, help='Training images to use')
    parser.add_argument('--test_samples_per_class', type=int, default=200, help='Test images per class')
    parser.add_argument('--lca_iterations', type=int, default=50, help='LCA iterations')
    parser.add_argument('--data_dir', type=str, default='./data', help='CIFAR-10 download directory')
    parser.add_argument('--seed', type=int, default=42)
    # Additional V1 configs to sweep
    parser.add_argument('--sweep', action='store_true', help='Run parameter sweep')
    return parser.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_cifar10(data_dir, train_samples, test_per_class):
    """Load CIFAR-10 with normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
    ])

    print("Loading CIFAR-10...")
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Subsample training set (just need patches for dictionary learning)
    train_indices = np.random.choice(len(trainset), min(train_samples, len(trainset)), replace=False)
    train_images = torch.stack([trainset[i][0] for i in train_indices])  # (N, 3, 32, 32)
    print(f"  Training images: {train_images.shape}")

    # Organize test set by class
    test_by_class = defaultdict(list)
    for img, label in testset:
        if len(test_by_class[label]) < test_per_class:
            test_by_class[label].append(img)

    test_images = {}
    for label in range(10):
        test_images[label] = torch.stack(test_by_class[label])
        print(f"  Test class {label} ({CIFAR10_CLASSES[label]}): {test_images[label].shape}")

    return train_images, test_images


def extract_training_patches(images, patch_size, stride, max_patches=200000):
    """Extract and normalize patches from training images."""
    all_patches = []
    for img in images:
        patches = extract_patches(img, size=patch_size, stride=stride)
        all_patches.append(patches)

    all_patches = torch.cat(all_patches, dim=0)

    # Subsample if too many
    if len(all_patches) > max_patches:
        indices = np.random.choice(len(all_patches), max_patches, replace=False)
        all_patches = all_patches[indices]

    # Normalize patches: zero mean, unit variance per patch
    all_patches = all_patches - all_patches.mean(dim=1, keepdim=True)
    std = all_patches.std(dim=1, keepdim=True)
    all_patches = all_patches / (std + 1e-8)

    print(f"  Training patches: {all_patches.shape}")
    return all_patches


# ============================================================================
# V1 TRAINING
# ============================================================================

def train_v1(model, patches, n_epochs, device, batch_size=256):
    """Train V1 dictionary on patches."""
    patches = patches.to(device)
    n_patches = len(patches)

    print(f"\nTraining V1 ({model.n_bases} bases, λ={model.lambda_sparse}) "
          f"for {n_epochs} epochs on {n_patches} patches...")

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n_patches)
        total_mse = 0
        n_batches = 0

        for start in range(0, n_patches, batch_size):
            end = min(start + batch_size, n_patches)
            batch = patches[perm[start:end]]

            # LCA inference
            codes = model.lca_inference(batch)

            # Hebbian dictionary learning
            mse = model.learn_batch(batch, codes)
            total_mse += mse
            n_batches += 1

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_batch = patches[:500]
            val_codes = model.lca_inference(val_batch)
            sparsity = model.get_sparsity(val_codes)
            capacity = model.get_effective_capacity()
            avg_mse = total_mse / n_batches
            print(f"  Epoch {epoch+1:3d}: MSE={avg_mse:.4f}, "
                  f"Sparsity={sparsity:.1%}, Capacity={capacity:.1%}")

    # Final stats
    val_codes = model.lca_inference(patches[:1000])
    final_sparsity = model.get_sparsity(val_codes)
    print(f"\nFinal V1: Sparsity={final_sparsity:.1%}")
    return final_sparsity


# ============================================================================
# CLUSTER GAP MEASUREMENT
# ============================================================================

def compute_image_codes(model, images_by_class, device, pool='max', stride=None):
    """Compute V1 sparse codes for all test images, grouped by class."""
    if stride is None:
        stride = model.patch_size  # non-overlapping

    codes_by_class = {}
    for label, images in images_by_class.items():
        images = images.to(device)
        with torch.no_grad():
            codes = model.forward(images, stride=stride, pool=pool)
        # Normalize codes for cosine similarity
        codes = F.normalize(codes, p=2, dim=1)
        codes_by_class[label] = codes.cpu()
        sparsity = model.get_sparsity(codes)
        print(f"  Class {label:2d} ({CIFAR10_CLASSES[label]:>10s}): "
              f"codes {codes.shape}, sparsity={sparsity:.1%}")

    return codes_by_class


def measure_cluster_gap(codes_by_class, n_classes=10):
    """
    Measure within-class vs between-class cosine similarity.

    Returns:
        cluster_gap: within_mean - between_mean
        within_mean: mean cosine similarity of same-class pairs
        between_mean: mean cosine similarity of different-class pairs
        per_class_stats: dict with per-class within/between similarities
    """
    within_sims = []
    between_sims = []
    per_class = {}

    classes = sorted(codes_by_class.keys())

    for c in classes:
        codes_c = codes_by_class[c]  # (n, dim)
        n = len(codes_c)

        # Within-class similarity (upper triangle of sim matrix)
        sim_within = codes_c @ codes_c.T
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        class_within = sim_within[mask].tolist()
        within_sims.extend(class_within)

        # Between-class similarity (against all other classes)
        class_between = []
        for c2 in classes:
            if c2 != c:
                codes_c2 = codes_by_class[c2]
                sim_between = codes_c @ codes_c2.T
                class_between.extend(sim_between.flatten().tolist())
        between_sims.extend(class_between)

        per_class[c] = {
            'within_mean': np.mean(class_within) if class_within else 0,
            'within_std': np.std(class_within) if class_within else 0,
            'between_mean': np.mean(class_between) if class_between else 0,
            'between_std': np.std(class_between) if class_between else 0,
        }

    within_mean = np.mean(within_sims)
    between_mean = np.mean(between_sims)
    cluster_gap = within_mean - between_mean

    return cluster_gap, within_mean, between_mean, per_class


def measure_pairwise_class_gaps(codes_by_class):
    """Measure cluster gap for each pair of classes (some pairs may be harder)."""
    classes = sorted(codes_by_class.keys())
    n_classes = len(classes)
    gap_matrix = np.zeros((n_classes, n_classes))

    for i, ci in enumerate(classes):
        codes_i = codes_by_class[ci]
        sim_within_i = codes_i @ codes_i.T
        mask = torch.triu(torch.ones(len(codes_i), len(codes_i), dtype=torch.bool), diagonal=1)
        within_i = sim_within_i[mask].mean().item()

        for j, cj in enumerate(classes):
            if i == j:
                gap_matrix[i][j] = 0
            else:
                codes_j = codes_by_class[cj]
                between_ij = (codes_i @ codes_j.T).mean().item()
                # Gap = how much more similar within-class is vs this specific between-class
                gap_matrix[i][j] = within_i - between_ij

    return gap_matrix


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_single_config(args, n_bases, patch_size, lambda_sparse, n_epochs):
    """Run one V1 config and measure cluster gap."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"\nDevice: {device}")

    # Load data
    train_images, test_images = load_cifar10(
        args.data_dir, args.train_samples, args.test_samples_per_class
    )

    # Extract training patches
    patches = extract_training_patches(
        train_images, patch_size=patch_size, stride=patch_size
    )

    # Create and train V1
    model = V1SparseCoding(
        n_bases=n_bases,
        patch_size=patch_size,
        n_channels=3,
        lambda_sparse=lambda_sparse,
        lca_iterations=args.lca_iterations,
    ).to(device)

    t0 = time.time()
    final_sparsity = train_v1(model, patches, n_epochs, device)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")

    # Compute codes for test images
    print(f"\nComputing test codes...")
    codes_by_class = compute_image_codes(model, test_images, device, pool='max')

    # Measure cluster gap
    print(f"\n{'='*60}")
    print(f"CLUSTER GAP ANALYSIS")
    print(f"{'='*60}")

    cluster_gap, within_mean, between_mean, per_class = measure_cluster_gap(codes_by_class)

    print(f"\nOverall:")
    print(f"  Within-class similarity:  {within_mean:.4f}")
    print(f"  Between-class similarity: {between_mean:.4f}")
    print(f"  Cluster gap:              {cluster_gap:.4f}")
    print(f"  V6 threshold:             {CLUSTER_GAP_THRESHOLD:.4f}")
    print(f"  Status: {'✓ PASS' if cluster_gap >= CLUSTER_GAP_THRESHOLD else '✗ FAIL'}")

    print(f"\nPer-class breakdown:")
    print(f"  {'Class':>12s} | {'Within':>8s} | {'Between':>8s} | {'Gap':>8s} | Status")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-------")
    for c in sorted(per_class.keys()):
        stats = per_class[c]
        gap = stats['within_mean'] - stats['between_mean']
        status = '✓' if gap >= CLUSTER_GAP_THRESHOLD else '✗'
        print(f"  {CIFAR10_CLASSES[c]:>12s} | {stats['within_mean']:8.4f} | "
              f"{stats['between_mean']:8.4f} | {gap:8.4f} | {status}")

    # Pairwise class gaps
    gap_matrix = measure_pairwise_class_gaps(codes_by_class)
    print(f"\nPairwise class gaps (hardest pairs):")
    pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            avg_gap = (gap_matrix[i][j] + gap_matrix[j][i]) / 2
            pairs.append((avg_gap, i, j))
    pairs.sort()

    print(f"  Hardest to distinguish:")
    for gap, i, j in pairs[:5]:
        print(f"    {CIFAR10_CLASSES[i]:>10s} vs {CIFAR10_CLASSES[j]:<10s}: gap = {gap:.4f}")
    print(f"  Easiest to distinguish:")
    for gap, i, j in pairs[-3:]:
        print(f"    {CIFAR10_CLASSES[i]:>10s} vs {CIFAR10_CLASSES[j]:<10s}: gap = {gap:.4f}")

    # Also measure: does sparsity correlate with cluster gap?
    # (more active neurons might give more discriminative codes)
    print(f"\n{'='*60}")
    print(f"V1 DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"  Bases: {n_bases}, Patch: {patch_size}×{patch_size}, λ_sparse: {lambda_sparse}")
    print(f"  Sparsity: {final_sparsity:.1%}")
    print(f"  Cluster gap: {cluster_gap:.4f}")

    if cluster_gap < 0.05:
        print(f"\n  DIAGNOSIS: V1 codes are essentially random w.r.t. categories.")
        print(f"  LCA learns local texture features (Gabor-like), not category-level")
        print(f"  structure. This is expected — V1 is a low-level feature extractor.")
        print(f"  Category clustering requires higher-level representations.")
    elif cluster_gap < CLUSTER_GAP_THRESHOLD:
        print(f"\n  DIAGNOSIS: V1 codes show SOME category structure, but insufficient")
        print(f"  for V6 binding. The gap ({cluster_gap:.4f}) is below threshold ({CLUSTER_GAP_THRESHOLD}).")
        print(f"  Options: (a) more V1 training, (b) different sparsity levels,")
        print(f"  (c) accept that V1→ATL needs an intermediate representation.")
    else:
        print(f"\n  DIAGNOSIS: V1 codes have sufficient category structure for V6 binding!")

    return {
        'n_bases': n_bases,
        'patch_size': patch_size,
        'lambda_sparse': lambda_sparse,
        'sparsity': final_sparsity,
        'cluster_gap': cluster_gap,
        'within_mean': within_mean,
        'between_mean': between_mean,
        'train_time': train_time,
    }


def run_sweep(args):
    """Sweep V1 hyperparameters to find best cluster gap."""
    configs = [
        # (n_bases, patch_size, lambda_sparse, n_epochs)
        (64,  8, 0.3, args.n_epochs),
        (64,  8, 0.5, args.n_epochs),
        (128, 8, 0.3, args.n_epochs),
        (128, 8, 0.5, args.n_epochs),
        (256, 8, 0.3, args.n_epochs),
        (256, 8, 0.5, args.n_epochs),
        # Lower sparsity = more active neurons = potentially more discriminative
        (128, 8, 0.1, args.n_epochs),
        (128, 8, 0.2, args.n_epochs),
    ]

    results = []
    for n_bases, patch_size, lambda_sparse, n_epochs in configs:
        print(f"\n{'#'*70}")
        print(f"CONFIG: bases={n_bases}, patch={patch_size}, λ={lambda_sparse}")
        print(f"{'#'*70}")
        result = run_single_config(args, n_bases, patch_size, lambda_sparse, n_epochs)
        results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print(f"SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Bases':>6s} | {'Patch':>5s} | {'λ':>5s} | {'Sparsity':>9s} | "
          f"{'Gap':>8s} | {'Within':>8s} | {'Between':>8s} | Status")
    print(f"{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-------")

    for r in sorted(results, key=lambda x: -x['cluster_gap']):
        status = '✓' if r['cluster_gap'] >= CLUSTER_GAP_THRESHOLD else '✗'
        print(f"{r['n_bases']:6d} | {r['patch_size']:5d} | {r['lambda_sparse']:5.2f} | "
              f"{r['sparsity']:9.1%} | {r['cluster_gap']:8.4f} | "
              f"{r['within_mean']:8.4f} | {r['between_mean']:8.4f} | {status}")

    best = max(results, key=lambda x: x['cluster_gap'])
    print(f"\nBest config: bases={best['n_bases']}, λ={best['lambda_sparse']}, "
          f"gap={best['cluster_gap']:.4f}")

    if best['cluster_gap'] < CLUSTER_GAP_THRESHOLD:
        print(f"\nNO CONFIG REACHES THRESHOLD ({CLUSTER_GAP_THRESHOLD}).")
        print(f"V1 LCA alone cannot produce category-level clustering on CIFAR-10.")
        print(f"This is architecturally significant — see DIAGNOSIS in single-config output.")
    else:
        print(f"\n✓ V6 BINDING IS VIABLE with this V1 configuration.")

    return results


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("MODAS: V1 LCA CLUSTER GAP ON REAL IMAGES (CIFAR-10)")
    print("=" * 70)
    print(f"\nQuestion: Does V1 produce codes with cluster gap ≥ {CLUSTER_GAP_THRESHOLD}?")
    print(f"If yes → V6 associative binding works with real V1 output")
    print(f"If no  → V1 needs changes or an intermediate layer is needed")

    if args.sweep:
        results = run_sweep(args)
    else:
        result = run_single_config(
            args, args.n_bases, args.patch_size, args.lambda_sparse, args.n_epochs
        )

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
