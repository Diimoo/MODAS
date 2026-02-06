#!/usr/bin/env python3
"""
MODAS Experiment: V2 Three-Factor Learning — Temporal Structure vs Unsupervised

THE CRITICAL QUESTION:
Can temporal category structure (same-category bursts) with scalar modulator
gating produce V2 codes with cluster gap ≥ 0.20, starting from V1 codes
that only achieve gap = 0.04?

DESIGN:
Two V2 competitive layers trained from identical initialization on identical
V1 codes from CIFAR-10:

  Condition A (Unsupervised):
    - Random data order
    - Standard competitive Hebbian learning
    - Control: can competitive learning alone amplify V1's weak 0.04 gap?

  Condition B (Three-Factor):
    - Same-category bursts (burst_size examples in a row from one category)
    - Scalar modulator (e.g. 3×) during bursts amplifies learning rate
    - Random interleave between bursts
    - The temporal structure carries category information
    - The modulator gates stronger learning during structured episodes

Neither condition has explicit labels on individual V2 units.
The only difference is data ordering and a scalar LR multiplier.

USAGE:
    cd /home/ahmed/Dokumente/MODAS
    source venv/bin/activate

    # Quick single comparison (~5-10 min on GPU)
    python experiments/experiment_v2_threefactor.py --device cuda --v2_epochs 30

    # Modulator sweep (~25-30 min on GPU)
    python experiments/experiment_v2_threefactor.py --device cuda --v2_epochs 30 --sweep_modulator
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    print("ERROR: torchvision required. Install with: pip install torchvision")
    sys.exit(1)

from modas.modules.v1_sparse_coding import V1SparseCoding, extract_patches

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
CLUSTER_GAP_THRESHOLD = 0.20


# ============================================================================
# V2 COMPETITIVE LAYER
# ============================================================================

class V2CompetitiveLayer:
    """
    V2 competitive layer with Hebbian prototype learning.
    
    Learns prototypes via competitive Hebbian: winning prototypes
    move toward the input. Three-factor variant scales the learning
    rate by an external scalar modulator.
    
    This is NOT an nn.Module — it's a simple buffer-based layer
    to keep things transparent and avoid autograd.
    """
    
    def __init__(self, input_dim, n_units, k=3, lr=0.05, device='cpu'):
        self.input_dim = input_dim
        self.n_units = n_units
        self.k = k
        self.lr = lr
        self.device = device
        
        # Initialize prototypes as random unit vectors
        self.prototypes = F.normalize(
            torch.randn(n_units, input_dim, device=device), dim=1
        )
        # Usage count for meta-plasticity (prevent dead units)
        self.usage = torch.zeros(n_units, device=device)
    
    def clone(self):
        """Create an identical copy (same initialization for fair comparison)."""
        other = V2CompetitiveLayer(
            self.input_dim, self.n_units, self.k, self.lr, self.device
        )
        other.prototypes = self.prototypes.clone()
        other.usage = self.usage.clone()
        return other
    
    def activate(self, x):
        """
        Sparse top-k activation.
        
        Args:
            x: (B, input_dim) normalized input
        Returns:
            (B, n_units) sparse activation pattern
        """
        x = F.normalize(x, p=2, dim=-1)
        sims = x @ self.prototypes.T  # (B, n_units)
        
        topk_vals, topk_idx = sims.topk(self.k, dim=-1)
        sparse_act = torch.zeros_like(sims)
        topk_softmax = F.softmax(topk_vals / 0.2, dim=-1)  # temperature=0.2
        sparse_act.scatter_(1, topk_idx, topk_softmax)
        
        return sparse_act
    
    def learn_step(self, x, modulator=1.0):
        """
        Competitive Hebbian learning step.
        
        Args:
            x: (B, input_dim) input batch
            modulator: scalar learning rate multiplier (three-factor signal)
        """
        with torch.no_grad():
            x = F.normalize(x, p=2, dim=-1)
            acts = self.activate(x)  # (B, n_units)
            
            # Weighted input: each prototype moves toward its winners
            # delta_p[j] = sum_i(act[i,j] * x[i]) / sum_i(act[i,j])
            act_sum = acts.sum(dim=0) + 1e-8  # (n_units,)
            weighted_x = acts.T @ x  # (n_units, input_dim)
            
            # Meta-plasticity: boost learning for underused units
            meta_lr = 1.0 / (1.0 + 0.01 * self.usage)
            
            effective_lr = self.lr * modulator * meta_lr
            
            # Update: move prototypes toward weighted input centroid
            delta = weighted_x / act_sum.unsqueeze(1) - self.prototypes
            self.prototypes = self.prototypes + effective_lr.unsqueeze(1) * delta
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
            
            # Track usage
            self.usage = self.usage + acts.sum(dim=0)
    
    def get_codes(self, x):
        """Get normalized activation codes for cluster gap measurement."""
        with torch.no_grad():
            acts = self.activate(x)
            return F.normalize(acts, p=2, dim=-1)


# ============================================================================
# DATA PIPELINE
# ============================================================================

def load_cifar10_with_labels(data_dir, train_samples, test_per_class):
    """Load CIFAR-10 with labels preserved."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    print("Loading CIFAR-10...")
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    # Training: sample with labels
    indices = np.random.choice(len(trainset), min(train_samples, len(trainset)), replace=False)
    train_images = torch.stack([trainset[i][0] for i in indices])
    train_labels = torch.tensor([trainset[i][1] for i in indices])
    print(f"  Training: {train_images.shape}, labels: {train_labels.shape}")
    
    # Test: organized by class
    test_by_class = defaultdict(list)
    for img, label in testset:
        if len(test_by_class[label]) < test_per_class:
            test_by_class[label].append(img)
    
    test_images = {}
    for label in range(10):
        test_images[label] = torch.stack(test_by_class[label])
    
    return train_images, train_labels, test_images


def compute_v1_codes(v1_model, images, device, pool='max'):
    """Compute V1 sparse codes for a batch of images."""
    images = images.to(device)
    with torch.no_grad():
        codes = v1_model.forward(images, stride=v1_model.patch_size, pool=pool)
    return codes


def train_v1_on_patches(images, device, n_bases=128, patch_size=8,
                        lambda_sparse=0.5, n_epochs=50, lca_iterations=50):
    """Train V1 and return the model."""
    # Extract patches
    all_patches = []
    for img in images:
        patches = extract_patches(img, size=patch_size, stride=patch_size)
        all_patches.append(patches)
    all_patches = torch.cat(all_patches, dim=0)
    
    if len(all_patches) > 200000:
        idx = np.random.choice(len(all_patches), 200000, replace=False)
        all_patches = all_patches[idx]
    
    # Normalize
    all_patches = all_patches - all_patches.mean(dim=1, keepdim=True)
    std = all_patches.std(dim=1, keepdim=True)
    all_patches = all_patches / (std + 1e-8)
    all_patches = all_patches.to(device)
    
    model = V1SparseCoding(
        n_bases=n_bases, patch_size=patch_size, n_channels=3,
        lambda_sparse=lambda_sparse, lca_iterations=lca_iterations,
    ).to(device)
    
    print(f"\nTraining V1 ({n_bases} bases, λ={lambda_sparse}, {n_epochs} epochs)...")
    n_patches = len(all_patches)
    for epoch in range(n_epochs):
        perm = torch.randperm(n_patches)
        for start in range(0, n_patches, 256):
            end = min(start + 256, n_patches)
            batch = all_patches[perm[start:end]]
            codes = model.lca_inference(batch)
            model.learn_batch(batch, codes)
        
        if (epoch + 1) % 10 == 0:
            val_codes = model.lca_inference(all_patches[:500])
            sp = model.get_sparsity(val_codes)
            print(f"  Epoch {epoch+1}: sparsity={sp:.1%}")
    
    print(f"  V1 training complete.")
    return model


# ============================================================================
# V2 TRAINING: UNSUPERVISED vs THREE-FACTOR
# ============================================================================

def train_v2_unsupervised(v2, codes, n_epochs, batch_size=64):
    """
    Condition A: Standard competitive Hebbian, random order.
    """
    n = len(codes)
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=codes.device)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = codes[perm[start:end]]
            v2.learn_step(batch, modulator=1.0)


def train_v2_threefactor(v2, codes, labels, n_epochs, burst_size=16,
                         modulator=3.0, batch_size=64):
    """
    Condition B: Same-category bursts with scalar modulator.
    
    Training protocol:
    1. Group codes by category
    2. Each "episode": pick a random category, sample burst_size examples
    3. Feed burst as a batch with modulator × learning rate
    4. Between bursts: feed random mixed batches with modulator=1.0
    
    This simulates: caregiver shows the child several dogs in a row
    ("look, a dog! another dog!"), then switches to something else.
    The scalar modulator represents heightened attention during
    structured episodes.
    """
    n = len(codes)
    unique_labels = labels.unique()
    n_classes = len(unique_labels)
    
    # Pre-group codes by class
    class_indices = {}
    for c in unique_labels:
        class_indices[c.item()] = torch.where(labels == c)[0]
    
    # Each epoch: alternate between category bursts and random batches
    for epoch in range(n_epochs):
        # Determine number of bursts per epoch
        # ~50% of data in bursts, ~50% random
        n_bursts = n // (2 * burst_size)
        
        # Phase 1: Category bursts (structured episodes)
        burst_order = torch.randint(0, n_classes, (n_bursts,))
        for bi in range(n_bursts):
            c = unique_labels[burst_order[bi]].item()
            idx = class_indices[c]
            # Sample burst_size examples from this category
            sample_idx = idx[torch.randint(0, len(idx), (burst_size,))]
            burst_batch = codes[sample_idx]
            v2.learn_step(burst_batch, modulator=modulator)
        
        # Phase 2: Random interleave (unstructured)
        perm = torch.randperm(n, device=codes.device)
        n_random = n // 2
        for start in range(0, n_random, batch_size):
            end = min(start + batch_size, n_random)
            batch = codes[perm[start:end]]
            v2.learn_step(batch, modulator=1.0)


# ============================================================================
# CLUSTER GAP MEASUREMENT
# ============================================================================

def measure_cluster_gap(codes_by_class):
    """Measure within-class vs between-class cosine similarity of V2 codes."""
    within_sims = []
    between_sims = []
    per_class = {}
    classes = sorted(codes_by_class.keys())
    
    for c in classes:
        codes_c = codes_by_class[c]
        n = len(codes_c)
        
        sim_within = codes_c @ codes_c.T
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        class_within = sim_within[mask].tolist()
        within_sims.extend(class_within)
        
        class_between = []
        for c2 in classes:
            if c2 != c:
                sim_b = codes_c @ codes_by_class[c2].T
                class_between.extend(sim_b.flatten().tolist())
        between_sims.extend(class_between)
        
        per_class[c] = {
            'within': np.mean(class_within) if class_within else 0,
            'between': np.mean(class_between) if class_between else 0,
        }
    
    w = np.mean(within_sims)
    b = np.mean(between_sims)
    return w - b, w, b, per_class


def get_v2_codes_by_class(v2, v1_model, test_images, device):
    """Compute V2 codes for test images, grouped by class."""
    codes_by_class = {}
    for label, images in test_images.items():
        v1_codes = compute_v1_codes(v1_model, images, device)
        v2_codes = v2.get_codes(v1_codes)
        codes_by_class[label] = v2_codes.cpu()
    return codes_by_class


# ============================================================================
# MAIN EXPERIMENTS
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='V2 Three-Factor Experiment')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--v1_epochs', type=int, default=50)
    parser.add_argument('--v2_epochs', type=int, default=30)
    parser.add_argument('--v2_units', type=int, default=64, help='Number of V2 prototypes')
    parser.add_argument('--v2_k', type=int, default=3, help='V2 top-k sparsity')
    parser.add_argument('--v2_lr', type=float, default=0.05, help='V2 learning rate')
    parser.add_argument('--burst_size', type=int, default=16, help='Category burst size')
    parser.add_argument('--modulator', type=float, default=3.0, help='Three-factor modulator strength')
    parser.add_argument('--train_samples', type=int, default=10000)
    parser.add_argument('--test_per_class', type=int, default=200)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sweep_modulator', action='store_true',
                        help='Sweep modulator values')
    parser.add_argument('--sweep_v2', action='store_true',
                        help='Sweep V2 architecture (units, k, lr)')
    return parser.parse_args()


def run_comparison(v1_model, train_codes, train_labels, test_images,
                   device, v2_units, v2_k, v2_lr, v2_epochs,
                   burst_size, modulator, seed=42):
    """Run one unsupervised vs three-factor comparison."""
    torch.manual_seed(seed)
    
    input_dim = train_codes.shape[1]
    
    # Create V2 layers with IDENTICAL initialization
    v2_unsup = V2CompetitiveLayer(input_dim, v2_units, k=v2_k, lr=v2_lr, device=device)
    v2_3fact = v2_unsup.clone()
    
    # Train unsupervised
    t0 = time.time()
    train_v2_unsupervised(v2_unsup, train_codes, v2_epochs)
    t_unsup = time.time() - t0
    
    # Train three-factor
    t0 = time.time()
    train_v2_threefactor(v2_3fact, train_codes, train_labels, v2_epochs,
                         burst_size=burst_size, modulator=modulator)
    t_3fact = time.time() - t0
    
    # Measure V1 baseline gap
    v1_codes_by_class = {}
    for label, images in test_images.items():
        v1_codes = compute_v1_codes(v1_model, images, device)
        v1_codes_by_class[label] = F.normalize(v1_codes, p=2, dim=-1).cpu()
    gap_v1, w_v1, b_v1, _ = measure_cluster_gap(v1_codes_by_class)
    
    # Measure V2 unsupervised gap
    codes_unsup = get_v2_codes_by_class(v2_unsup, v1_model, test_images, device)
    gap_unsup, w_unsup, b_unsup, pc_unsup = measure_cluster_gap(codes_unsup)
    
    # Measure V2 three-factor gap
    codes_3fact = get_v2_codes_by_class(v2_3fact, v1_model, test_images, device)
    gap_3fact, w_3fact, b_3fact, pc_3fact = measure_cluster_gap(codes_3fact)
    
    return {
        'v1_gap': gap_v1, 'v1_within': w_v1, 'v1_between': b_v1,
        'unsup_gap': gap_unsup, 'unsup_within': w_unsup, 'unsup_between': b_unsup,
        '3fact_gap': gap_3fact, '3fact_within': w_3fact, '3fact_between': b_3fact,
        'unsup_time': t_unsup, '3fact_time': t_3fact,
        'per_class_unsup': pc_unsup, 'per_class_3fact': pc_3fact,
        'modulator': modulator, 'v2_units': v2_units, 'v2_k': v2_k,
        'burst_size': burst_size,
    }


def print_result(r, verbose=True):
    """Print comparison result."""
    ratio = r['3fact_gap'] / max(r['unsup_gap'], 1e-6)
    status_3f = '✓' if r['3fact_gap'] >= CLUSTER_GAP_THRESHOLD else '✗'
    
    print(f"\n  V1 baseline:   gap={r['v1_gap']:.4f}  (within={r['v1_within']:.4f}, between={r['v1_between']:.4f})")
    print(f"  V2 unsupervised: gap={r['unsup_gap']:.4f}  (within={r['unsup_within']:.4f}, between={r['unsup_between']:.4f})  [{r['unsup_time']:.1f}s]")
    print(f"  V2 three-factor: gap={r['3fact_gap']:.4f}  (within={r['3fact_within']:.4f}, between={r['3fact_between']:.4f})  [{r['3fact_time']:.1f}s]  {status_3f}")
    print(f"  Ratio (3-factor / unsup): {ratio:.2f}×")
    print(f"  V6 threshold: {CLUSTER_GAP_THRESHOLD}")
    
    if verbose:
        print(f"\n  Per-class gaps:")
        print(f"  {'Class':>12s} | {'Unsup':>8s} | {'3-Factor':>8s} | {'Δ':>8s}")
        print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for c in sorted(r['per_class_unsup'].keys()):
            u = r['per_class_unsup'][c]
            t = r['per_class_3fact'][c]
            g_u = u['within'] - u['between']
            g_t = t['within'] - t['between']
            print(f"  {CIFAR10_CLASSES[c]:>12s} | {g_u:8.4f} | {g_t:8.4f} | {g_t - g_u:+8.4f}")


def exp_single(args, v1_model, train_codes, train_labels, test_images, device):
    """Single comparison with default parameters."""
    print("\n" + "=" * 70)
    print("EXP 1: UNSUPERVISED vs THREE-FACTOR (single config)")
    print(f"  V2: {args.v2_units} units, k={args.v2_k}, lr={args.v2_lr}")
    print(f"  Three-factor: burst={args.burst_size}, modulator={args.modulator}×")
    print("=" * 70)
    
    r = run_comparison(
        v1_model, train_codes, train_labels, test_images, device,
        v2_units=args.v2_units, v2_k=args.v2_k, v2_lr=args.v2_lr,
        v2_epochs=args.v2_epochs, burst_size=args.burst_size,
        modulator=args.modulator, seed=args.seed,
    )
    print_result(r)
    return r


def exp_modulator_sweep(args, v1_model, train_codes, train_labels, test_images, device):
    """Sweep modulator strength."""
    print("\n" + "=" * 70)
    print("EXP 2: MODULATOR STRENGTH SWEEP")
    print(f"  V2: {args.v2_units} units, k={args.v2_k}, burst={args.burst_size}")
    print("=" * 70)
    
    modulators = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]
    results = []
    
    for mod in modulators:
        r = run_comparison(
            v1_model, train_codes, train_labels, test_images, device,
            v2_units=args.v2_units, v2_k=args.v2_k, v2_lr=args.v2_lr,
            v2_epochs=args.v2_epochs, burst_size=args.burst_size,
            modulator=mod, seed=args.seed,
        )
        results.append(r)
    
    print(f"\n{'Mod':>6s} | {'Unsup Gap':>10s} | {'3F Gap':>10s} | {'Ratio':>6s} | Status")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}-+-------")
    for r in results:
        ratio = r['3fact_gap'] / max(r['unsup_gap'], 1e-6)
        status = '✓' if r['3fact_gap'] >= CLUSTER_GAP_THRESHOLD else '✗'
        print(f"{r['modulator']:6.1f} | {r['unsup_gap']:10.4f} | {r['3fact_gap']:10.4f} | {ratio:6.2f}× | {status}")
    
    # Key question: does modulator=1.0 (just temporal structure, no gating)
    # give the same result as higher modulators?
    base = results[0]  # modulator=1.0
    print(f"\n  Temporal structure alone (mod=1.0): gap={base['3fact_gap']:.4f}")
    print(f"  Best modulator: {max(results, key=lambda x: x['3fact_gap'])['modulator']:.1f}×, "
          f"gap={max(results, key=lambda x: x['3fact_gap'])['3fact_gap']:.4f}")
    
    return results


def exp_v2_sweep(args, v1_model, train_codes, train_labels, test_images, device):
    """Sweep V2 architecture parameters."""
    print("\n" + "=" * 70)
    print("EXP 3: V2 ARCHITECTURE SWEEP (modulator=3.0)")
    print("=" * 70)
    
    configs = [
        # (units, k, lr, burst_size)
        (32,  2, 0.05, 16),
        (32,  3, 0.05, 16),
        (64,  2, 0.05, 16),
        (64,  3, 0.05, 16),
        (64,  5, 0.05, 16),
        (128, 3, 0.05, 16),
        (128, 5, 0.05, 16),
        # Different burst sizes
        (64,  3, 0.05, 8),
        (64,  3, 0.05, 32),
        (64,  3, 0.05, 64),
        # Different learning rates
        (64,  3, 0.02, 16),
        (64,  3, 0.10, 16),
    ]
    
    print(f"\n{'Config':<25s} | {'Unsup':>8s} | {'3-Factor':>8s} | {'Ratio':>6s} | Status")
    print(f"{'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-------")
    
    best_gap, best_cfg = -1, ""
    for units, k, lr, burst in configs:
        r = run_comparison(
            v1_model, train_codes, train_labels, test_images, device,
            v2_units=units, v2_k=k, v2_lr=lr,
            v2_epochs=args.v2_epochs, burst_size=burst,
            modulator=3.0, seed=args.seed,
        )
        tag = f"u={units},k={k},lr={lr},b={burst}"
        ratio = r['3fact_gap'] / max(r['unsup_gap'], 1e-6)
        status = '✓' if r['3fact_gap'] >= CLUSTER_GAP_THRESHOLD else '✗'
        print(f"{tag:<25s} | {r['unsup_gap']:8.4f} | {r['3fact_gap']:8.4f} | {ratio:6.2f}× | {status}")
        
        if r['3fact_gap'] > best_gap:
            best_gap = r['3fact_gap']
            best_cfg = tag
    
    print(f"\nBest: {best_cfg} → gap={best_gap:.4f}")
    if best_gap < CLUSTER_GAP_THRESHOLD:
        print(f"NO CONFIG REACHES THRESHOLD ({CLUSTER_GAP_THRESHOLD}).")
    else:
        print(f"✓ THRESHOLD REACHED.")


def exp_multiple_seeds(args, v1_model, train_codes, train_labels, test_images, device):
    """Run best config across multiple seeds for statistical significance."""
    print("\n" + "=" * 70)
    print("EXP 4: MULTI-SEED STABILITY (5 seeds)")
    print(f"  V2: {args.v2_units} units, k={args.v2_k}, modulator={args.modulator}")
    print("=" * 70)
    
    unsup_gaps, threef_gaps = [], []
    for seed in [42, 123, 456, 789, 1337]:
        r = run_comparison(
            v1_model, train_codes, train_labels, test_images, device,
            v2_units=args.v2_units, v2_k=args.v2_k, v2_lr=args.v2_lr,
            v2_epochs=args.v2_epochs, burst_size=args.burst_size,
            modulator=args.modulator, seed=seed,
        )
        unsup_gaps.append(r['unsup_gap'])
        threef_gaps.append(r['3fact_gap'])
        print(f"  Seed {seed}: unsup={r['unsup_gap']:.4f}, 3-factor={r['3fact_gap']:.4f}")
    
    print(f"\n  Unsupervised:  {np.mean(unsup_gaps):.4f} ± {np.std(unsup_gaps):.4f}")
    print(f"  Three-factor:  {np.mean(threef_gaps):.4f} ± {np.std(threef_gaps):.4f}")
    ratio = np.mean(threef_gaps) / max(np.mean(unsup_gaps), 1e-6)
    print(f"  Mean ratio: {ratio:.2f}×")
    
    # Simple significance test
    diff = np.array(threef_gaps) - np.array(unsup_gaps)
    t_stat = np.mean(diff) / (np.std(diff) / np.sqrt(len(diff)) + 1e-8)
    print(f"  Paired difference: {np.mean(diff):.4f} ± {np.std(diff):.4f}, t={t_stat:.2f}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    
    print("=" * 70)
    print("MODAS: V2 THREE-FACTOR LEARNING EXPERIMENT")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"\nQuestion: Does temporal category structure + scalar modulator")
    print(f"produce V2 codes with cluster gap ≥ {CLUSTER_GAP_THRESHOLD}?")
    
    # Step 1: Load data
    train_images, train_labels, test_images = load_cifar10_with_labels(
        args.data_dir, args.train_samples, args.test_per_class
    )
    
    # Step 2: Train V1
    v1_model = train_v1_on_patches(train_images, device, n_epochs=args.v1_epochs)
    
    # Step 3: Compute V1 codes for all training images
    print("\nComputing V1 codes for training images...")
    train_codes = compute_v1_codes(v1_model, train_images, device)
    train_labels = train_labels.to(device)
    print(f"  V1 codes: {train_codes.shape}")
    
    # Step 4: Run experiments
    exp_single(args, v1_model, train_codes, train_labels, test_images, device)
    
    if args.sweep_modulator:
        exp_modulator_sweep(args, v1_model, train_codes, train_labels, test_images, device)
    
    if args.sweep_v2:
        exp_v2_sweep(args, v1_model, train_codes, train_labels, test_images, device)
    
    # Always run multi-seed for the default config
    exp_multiple_seeds(args, v1_model, train_codes, train_labels, test_images, device)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
