# MODAS - Modular Developmental Architecture for Semantics

**Train modules separately with proven methods, then bind them with local learning rules.**

## Overview

MODAS is a biologically-inspired architecture for multimodal semantic binding. It consists of:

- **V1 Visual Cortex**: Sparse coding via Locally Competitive Algorithm (LCA)
- **A1 Auditory Cortex**: Temporal sparse coding for audio features
- **Language Encoder**: Word2Vec-based semantic embeddings
- **ATL Semantic Hub**: Contrastive Hebbian Learning for cross-modal binding

## Success Metrics

| Module | Metric | Target |
|--------|--------|--------|
| V1 | Gabor filter emergence | > 0.6 |
| V1 | Sparsity | < 15% |
| A1 | Temporal filters | Emerge |
| A1 | Sparsity | < 15% |
| ATL | Discrimination | > 0.15 |
| Trimodal | Cross-modal retrieval | > 60% |

## Installation

```bash
# Create conda environment
conda create -n modas python=3.10
conda activate modas

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Project Structure

```
modas/
├── modas/
│   ├── modules/          # Core neural modules
│   │   ├── v1_sparse_coding.py
│   │   ├── a1_sparse_coding.py
│   │   ├── language_encoder.py
│   │   └── atl_semantic_hub.py
│   ├── training/         # Training scripts
│   ├── evaluation/       # Validation scripts
│   ├── data/            # Dataset loaders
│   └── utils/           # Utilities
├── experiments/         # Experiment outputs
├── tests/              # Unit tests
├── notebooks/          # Jupyter notebooks
└── docs/               # Documentation
```

## Quick Start

```python
from modas.modules import V1SparseCoding, A1SparseCoding, LanguageEncoder, ATLSemanticHub

# Initialize modules
v1 = V1SparseCoding()
a1 = A1SparseCoding()
lang = LanguageEncoder()
atl = ATLSemanticHub()

# Process inputs
vis_code = v1.forward(image_patch)
aud_code = a1.forward(audio_segment)
lang_emb = lang.forward("red circle")

# Bind in ATL
similarity, margin = atl.bind(vis_code, lang_emb)
```

## Training Phases

1. **Phase 1**: Train V1 sparse coding on natural images
2. **Phase 2**: Train A1 sparse coding on speech audio
3. **Phase 3**: Load/train language encoder
4. **Phase 4**: Train ATL binding (critical test)
5. **Phase 5**: Trimodal binding (optional)

## Key Lessons Applied

### From MBM:
- Meta-plasticity prevents collapse
- Usage tracking for capacity management
- Unit-norm constraints for stability

### From CHPL:
- NO softmax (creates artifacts)
- Temporal memory for hard negatives
- Contrastive margin for discrimination
- Three-factor Hebbian learning

## License

MIT License
