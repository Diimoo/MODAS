# MODAS Training Protocol

## Overview

MODAS follows a phased training approach:
1. Train sensory modules independently with proven methods
2. Validate each module before proceeding
3. Train ATL binding (the critical test)
4. Optional: Extend to trimodal binding

## Phase 1: V1 Visual Cortex

### Goal
Learn sparse, Gabor-like filters from natural images.

### Method
Locally Competitive Algorithm (LCA) sparse coding

### Training Steps

```bash
# 1. Prepare dataset (ImageNet subset or CIFAR)
python scripts/prepare_images.py --source imagenet --output data/images

# 2. Train V1
python -m modas.training.train_v1 \
    --image_dir data/images \
    --n_epochs 100 \
    --batch_size 64 \
    --n_bases 128 \
    --patch_size 16 \
    --device cuda

# 3. Validate
python -m modas.evaluation.validate_v1 \
    --model_path experiments/phase1_v1/checkpoints/v1_final.pt \
    --save_dir experiments/phase1_v1/validation
```

### Success Criteria
| Metric | Target | Description |
|--------|--------|-------------|
| Gabor Score | > 0.6 | Filters resemble oriented edges |
| Sparsity | > 85% | < 15% neurons active |
| Cross-similarity | < 0.5 | Different images → different codes |

### Checkpoints
- **PASS**: Continue to Phase 2
- **FAIL**: Debug V1 (this is foundational - must work)

---

## Phase 2: A1 Auditory Cortex

### Goal
Learn sparse temporal filters from speech audio.

### Method
Temporal sparse coding (1D LCA on spectrograms)

### Training Steps

```bash
# 1. Prepare audio dataset (LibriSpeech or similar)
python scripts/prepare_audio.py --source librispeech --output data/audio

# 2. Train A1
python -m modas.training.train_a1 \
    --audio_dir data/audio \
    --n_epochs 100 \
    --batch_size 64 \
    --n_bases 128 \
    --segment_length 3200 \
    --device cuda

# 3. Validate
python -m modas.evaluation.validate_a1 \
    --model_path experiments/phase2_a1/checkpoints/a1_final.pt \
    --save_dir experiments/phase2_a1/validation
```

### Success Criteria
| Metric | Target | Description |
|--------|--------|-------------|
| Temporal Score | > 0.3 | Filters show temporal structure |
| Sparsity | > 85% | < 15% neurons active |
| Cross-similarity | < 0.5 | Different sounds → different codes |

### Checkpoints
- **PASS**: Continue to Phase 4 (ATL)
- **FAIL**: Skip audio, proceed with V1↔Language only

---

## Phase 3: Language Encoder

### Goal
Obtain semantic word embeddings.

### Method
Pretrained Word2Vec with random projection

### Steps

```python
from modas.modules import LanguageEncoder

# Load pretrained (downloads automatically)
lang = LanguageEncoder(output_dim=128, load_pretrained=True)

# Test
emb = lang.forward("red circle")
print(f"Embedding shape: {emb.shape}")  # (128,)
```

### No Training Needed
Just load and use. Alternatively, train HebbianLanguageEncoder from scratch.

---

## Phase 4: ATL Binding (CRITICAL TEST)

### Goal
Learn cross-modal semantic binding.

### Method
Contrastive Hebbian Learning (CHL)

### Training Steps

```bash
# 1. Prepare multimodal dataset (COCO captions or synthetic)
python scripts/prepare_multimodal.py --output data/multimodal

# 2. Train ATL
python -m modas.training.train_atl \
    --v1_path experiments/phase1_v1/checkpoints/v1_final.pt \
    --data_source data/multimodal \
    --n_epochs 200 \
    --n_prototypes 200 \
    --device cuda

# 3. Validate (THE CRITICAL TEST)
python -m modas.evaluation.validate_atl \
    --atl_path experiments/phase4_atl_binding/checkpoints/atl_final.pt \
    --v1_path experiments/phase1_v1/checkpoints/v1_final.pt \
    --save_dir experiments/phase4_atl_binding/validation
```

### Success Criteria

| Discrimination | Status | Action |
|----------------|--------|--------|
| > 0.2 | EXCELLENT | Publish, continue to trimodal |
| 0.15 - 0.2 | GOOD | Publish, continue to trimodal |
| 0.1 - 0.15 | MARGINAL | Analyze, try improvements |
| < 0.1 | FAIL | Document negative result, STOP |

### Key Metrics
- **Discrimination**: mean(matched) - mean(mismatched)
- **Improvement over random**: Should be > 0.05
- **Swap generalization**: Test compositional understanding

---

## Phase 5: Trimodal Binding (Optional)

### Prerequisite
Phase 4 must achieve GOOD or EXCELLENT status.

### Goal
Bind all three modalities (V1 + A1 + Language)

### Training Steps

```bash
python -m modas.training.train_atl \
    --v1_path experiments/phase1_v1/checkpoints/v1_final.pt \
    --a1_path experiments/phase2_a1/checkpoints/a1_final.pt \
    --data_source data/multimodal_with_audio \
    --n_epochs 200 \
    --device cuda
```

### Success Criteria
- Cross-modal retrieval > 60%
- Audio-visual-language alignment

---

## Monitoring During Training

### Key Metrics to Track

```python
# Every N steps:
print(f"Similarity: {similarity:.3f}")
print(f"Margin: {margin:.3f}")
print(f"Capacity: {capacity:.2%}")

# Every epoch:
print(f"Discrimination: {disc:.3f}")
```

### Warning Signs
- **Capacity < 50%**: Prototype collapse (increase meta-plasticity)
- **Similarity always ~0.7**: Softmax artifact (ensure using sigmoid)
- **Margin always negative**: Learning signal wrong direction
- **No improvement over random**: Model not learning

### Visualization Commands

```bash
# Plot V1 filters
python -m modas.utils.visualization --plot_filters v1 \
    --model_path experiments/phase1_v1/checkpoints/v1_final.pt

# Plot training curves
python -m modas.utils.visualization --plot_training \
    --log_dir experiments/phase4_atl_binding/logs

# Plot discrimination results
python -m modas.utils.visualization --plot_discrimination \
    --results_path experiments/phase4_atl_binding/validation/results.json
```

---

## Troubleshooting

### V1 Not Learning Gabors
1. Check patch normalization (zero mean, unit variance)
2. Increase LCA iterations (try 100)
3. Adjust sparsity penalty (try range 0.05-0.2)
4. Ensure diverse training images

### A1 Not Learning Temporal Filters
1. Check spectrogram computation
2. Adjust segment length
3. Try different audio preprocessing

### ATL Discrimination Low
1. Check that V1/A1 codes are diverse
2. Increase memory buffer size
3. Add more hard negatives
4. Increase training epochs
5. Check meta-plasticity is active

### Capacity Collapse
1. Increase meta_beta (try 0.9999)
2. Reset usage counts periodically
3. Add regularization to encourage diversity

---

## Experiment Logging

Use wandb or tensorboard:

```python
import wandb

wandb.init(project="modas", name="atl_binding_v1")

# Log metrics
wandb.log({
    "discrimination": disc,
    "similarity": sim,
    "margin": margin,
    "capacity": capacity,
})

# Log visualizations
wandb.log({"filters": wandb.Image(filter_plot)})
```
