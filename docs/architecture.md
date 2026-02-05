# MODAS Architecture Specification

## Overview

MODAS (Modular Developmental Architecture for Semantics) is a biologically-inspired system for multimodal semantic binding. It trains separate sensory modules with proven sparse coding methods, then binds them using local Hebbian learning rules.

## Core Principle

> "Train modules separately with proven methods, then bind them with local learning rules. If binding fails, we know it's the binding mechanism, not the features."

## System Architecture

```
INPUT LAYER (Raw sensory data)
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   IMAGES    │  │    AUDIO    │  │    TEXT     │
│  (224×224)  │  │ (16kHz wav) │  │  (strings)  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
PROCESSING MODULES (Sparse coding, local learning)
┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
│     V1      │  │     A1      │  │  LANGUAGE   │
│  Sparse     │  │  Sparse     │  │  Embedding  │
│  Coding     │  │  Coding     │  │  (128-dim)  │
│  (128-dim)  │  │  (128-dim)  │  │             │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────┬───────┴────────┬───────┘
                │                │
SEMANTIC HUB (Multimodal binding)
         ┌──────▼────────────────▼──────┐
         │          ATL                 │
         │  (Anterior Temporal Lobe)    │
         │  - Contrastive Hebbian       │
         │  - Temporal memory           │
         │  - Meta-plasticity           │
         │  (200 prototypes × 128-dim)  │
         └──────────────────────────────┘
```

## Module Specifications

### Module 1: V1 Visual Cortex

**Function:** Transform raw images into sparse, discriminative features

**Method:** Sparse Coding via Locally Competitive Algorithm (LCA)

**Parameters:**
- 128 dictionary bases
- 16×16×3 patch size (768 input dimensions)
- λ_sparse = 0.1 (sparsity penalty)
- τ = 10.0 (LCA time constant)
- η = 0.01 (learning rate)
- 50 LCA iterations

**Output:** 128-dimensional sparse code

**Success Criteria:**
- Gabor score > 0.6 (filters resemble oriented edges)
- Sparsity > 85% (< 15% neurons active)
- Cross-similarity < 0.5 (different images → different codes)

### Module 2: A1 Auditory Cortex

**Function:** Transform audio into sparse temporal features

**Method:** Temporal Sparse Coding (1D variant of V1)

**Parameters:**
- 128 dictionary bases
- 200ms segments @ 16kHz (3200 samples)
- Spectrogram: 512 FFT, 160 hop length
- Same LCA parameters as V1

**Output:** 128-dimensional sparse code

**Success Criteria:**
- Temporal structure score > 0.3
- Sparsity > 85%
- Different sounds → different codes

### Module 3: Language Encoder

**Function:** Map words to dense semantic vectors

**Method:** Pretrained Word2Vec with random projection

**Parameters:**
- 300-dim Word2Vec → 128-dim projection
- Johnson-Lindenstrauss random projection
- L2 normalized output

**Output:** 128-dimensional dense embedding

### Module 4: ATL Semantic Hub

**Function:** Bind V1, A1, Language into unified semantic space

**Method:** Contrastive Hebbian Learning (CHL)

**Parameters:**
- 200 semantic prototypes × 128 dimensions
- Temperature τ = 0.2 (for sigmoid activation)
- Base learning rate η = 0.01
- Memory buffer size = 100
- 20 negative samples per update

**Key Design Choices (from lessons learned):**
- NO softmax (creates artifacts) → use sigmoid
- Temporal memory for hard negatives
- Contrastive margin for discrimination
- Three-factor Hebbian (pre × post × modulator)
- Meta-plasticity to prevent collapse

**Success Criteria:**
- Discrimination > 0.15 (bio-plausible target)
- Discrimination > 0.2 (excellent)

## Interfaces

| From | To | Dimension | Type |
|------|-----|-----------|------|
| V1 | ATL | 128 | Sparse |
| A1 | ATL | 128 | Sparse |
| Language | ATL | 128 | Dense |
| ATL | Output | 200 | Activation |

## Training Protocol

### Phase 1: V1 (Visual)
1. Train on natural images (ImageNet subset)
2. Extract 16×16 patches
3. Run LCA inference + Hebbian learning
4. Validate: Gabor emergence, sparsity, discrimination

### Phase 2: A1 (Auditory)
1. Train on speech audio (LibriSpeech)
2. Compute spectrograms, run LCA
3. Validate: temporal filters, sparsity

### Phase 3: Language
1. Load pretrained Word2Vec
2. Create projection layer
3. No training needed

### Phase 4: ATL Binding (CRITICAL)
1. Freeze V1, A1, Language modules
2. Train ATL with Contrastive Hebbian Learning
3. Validate discrimination score
4. **Decision point:** Continue if > 0.15, stop if < 0.1

### Phase 5: Trimodal (Optional)
1. Only if Phase 4 succeeds
2. Add audio to binding
3. Test cross-modal retrieval

## Lessons Applied

### From MBM (Memory-Based Model):
- ✓ Meta-plasticity prevents collapse
- ✓ Usage tracking for capacity management
- ✓ Unit-norm constraints for stability
- ✗ Trace decay (hurt performance)

### From CHPL (Compositional Hebbian Prototype Learning):
- ✓ Sigmoid activation (NOT softmax)
- ✓ Temporal memory for hard negatives
- ✓ Contrastive margin
- ✓ Three-factor Hebbian learning
- ✗ Winner-takes-all (too sparse)
- ✗ Pure Hebbian without modulation
