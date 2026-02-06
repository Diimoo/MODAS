"""
MODAS Neural Modules

- V1SparseCoding: Visual cortex sparse coding via LCA
- A1SparseCoding: Auditory cortex temporal sparse coding
- LanguageEncoder: Word2Vec-based language embeddings
- ATLSemanticHub: Cross-modal binding via Contrastive Hebbian Learning
"""

from .v1_sparse_coding import V1SparseCoding
from .a1_sparse_coding import A1SparseCoding
from .language_encoder import LanguageEncoder, HebbianLanguageEncoder
from .atl_semantic_hub import ATLSemanticHub
from .atl_semantic_hub_v2 import ATLSemanticHubV2
from .atl_semantic_hub_v3 import ATLSemanticHubV3
from .atl_semantic_hub_v4 import ATLSemanticHubV4

__all__ = [
    'V1SparseCoding',
    'A1SparseCoding', 
    'LanguageEncoder',
    'HebbianLanguageEncoder',
    'ATLSemanticHub',
    'ATLSemanticHubV2',
    'ATLSemanticHubV3',
    'ATLSemanticHubV4',
]
