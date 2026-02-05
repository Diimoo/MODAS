"""
MODAS Neural Modules

- V1SparseCoding: Visual cortex sparse coding via LCA
- A1SparseCoding: Auditory cortex temporal sparse coding
- LanguageEncoder: Word2Vec-based language embeddings
- ATLSemanticHub: Cross-modal binding via Contrastive Hebbian Learning
"""

from modas.modules.v1_sparse_coding import V1SparseCoding
from modas.modules.a1_sparse_coding import A1SparseCoding
from modas.modules.language_encoder import LanguageEncoder, HebbianLanguageEncoder
from modas.modules.atl_semantic_hub import ATLSemanticHub

__all__ = [
    "V1SparseCoding",
    "A1SparseCoding",
    "LanguageEncoder",
    "HebbianLanguageEncoder",
    "ATLSemanticHub",
]
