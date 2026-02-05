"""
MODAS - Modular Developmental Architecture for Semantics

A biologically-inspired architecture for multimodal semantic binding.
"""

__version__ = "0.1.0"

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
