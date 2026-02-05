"""
Language Encoder Module

Maps words/text to dense semantic vectors.
Two implementations:
1. LanguageEncoder: Uses pretrained Word2Vec (practical choice)
2. HebbianLanguageEncoder: Bio-plausible co-occurrence learning

Key features:
- Projection to ATL-compatible dimension (128)
- L2 normalized outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Union
from collections import defaultdict


class LanguageEncoder(nn.Module):
    """
    Language encoder using pretrained Word2Vec embeddings.
    
    Projects 300-dim Word2Vec to 128-dim for ATL interface.
    Uses random projection (Johnson-Lindenstrauss lemma).
    
    Args:
        output_dim: Output embedding dimension (default: 128)
        w2v_dim: Word2Vec dimension (default: 300)
        model_name: Gensim model name (default: 'word2vec-google-news-300')
        load_pretrained: Whether to load pretrained model (default: True)
    """
    
    def __init__(
        self,
        output_dim: int = 128,
        w2v_dim: int = 300,
        model_name: str = 'word2vec-google-news-300',
        load_pretrained: bool = True,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.w2v_dim = w2v_dim
        self.model_name = model_name
        
        # Random projection matrix (Johnson-Lindenstrauss)
        self.register_buffer(
            'projection',
            torch.randn(output_dim, w2v_dim) / np.sqrt(w2v_dim)
        )
        
        # Word2Vec model (loaded lazily)
        self.w2v = None
        self.w2v_loaded = False
        
        if load_pretrained:
            self._load_w2v()
    
    def _load_w2v(self):
        """Load pretrained Word2Vec model."""
        if self.w2v_loaded:
            return
        
        try:
            import gensim.downloader as api
            print(f"Loading Word2Vec model: {self.model_name}...")
            self.w2v = api.load(self.model_name)
            self.w2v_loaded = True
            print(f"Loaded {len(self.w2v)} word vectors.")
        except Exception as e:
            print(f"Warning: Could not load Word2Vec model: {e}")
            print("Using random embeddings instead.")
            self.w2v = None
            self.w2v_loaded = True
    
    def get_word_embedding(self, word: str) -> Optional[torch.Tensor]:
        """Get Word2Vec embedding for a single word."""
        if not self.w2v_loaded:
            self._load_w2v()
        
        if self.w2v is None:
            # Random embedding fallback
            return torch.randn(self.w2v_dim)
        
        word_lower = word.lower()
        if word_lower in self.w2v:
            return torch.tensor(self.w2v[word_lower], dtype=torch.float32)
        
        # Try original case
        if word in self.w2v:
            return torch.tensor(self.w2v[word], dtype=torch.float32)
        
        return None  # OOV
    
    def forward(
        self,
        text: Union[str, List[str]],
        pooling: str = 'mean'
    ) -> torch.Tensor:
        """
        Encode text to semantic embedding.
        
        Args:
            text: Input string or list of strings
            pooling: Pooling method for multi-word ('mean', 'max', 'first')
        
        Returns:
            Embedding (output_dim,) or (batch, output_dim) for list input
        """
        if isinstance(text, str):
            return self._encode_single(text, pooling)
        else:
            embeddings = [self._encode_single(t, pooling) for t in text]
            return torch.stack(embeddings)
    
    def _encode_single(self, text: str, pooling: str = 'mean') -> torch.Tensor:
        """Encode a single text string."""
        # Tokenize
        words = text.lower().split()
        
        # Get embeddings for each word
        embeddings = []
        for word in words:
            emb = self.get_word_embedding(word)
            if emb is not None:
                embeddings.append(emb)
        
        if len(embeddings) == 0:
            # OOV handling: return zero vector
            return torch.zeros(self.output_dim)
        
        # Stack embeddings
        emb_tensor = torch.stack(embeddings)  # (n_words, w2v_dim)
        
        # Pool
        if pooling == 'mean':
            pooled = emb_tensor.mean(dim=0)
        elif pooling == 'max':
            pooled = emb_tensor.max(dim=0)[0]
        elif pooling == 'first':
            pooled = emb_tensor[0]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # Project to output dimension
        device = self.projection.device
        pooled = pooled.to(device)
        projected = self.projection @ pooled
        
        # L2 normalize
        return F.normalize(projected, p=2, dim=0)
    
    def encode_words(self, words: List[str]) -> torch.Tensor:
        """
        Encode a list of individual words.
        
        Args:
            words: List of words
        
        Returns:
            Embeddings (n_words, output_dim)
        """
        embeddings = []
        for word in words:
            emb = self.get_word_embedding(word)
            if emb is not None:
                # Project and normalize
                device = self.projection.device
                emb = emb.to(device)
                proj = self.projection @ emb
                proj = F.normalize(proj, p=2, dim=0)
                embeddings.append(proj)
            else:
                # OOV: random embedding
                embeddings.append(torch.randn(self.output_dim))
        
        return torch.stack(embeddings)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.forward(text1)
        emb2 = self.forward(text2)
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


class HebbianLanguageEncoder(nn.Module):
    """
    Bio-plausible language encoder using Hebbian co-occurrence learning.
    
    Learns embeddings from text corpus via skip-gram style co-occurrence
    with Hebbian updates. 100% local learning rules.
    
    Args:
        vocab_size: Maximum vocabulary size (default: 10000)
        embedding_dim: Embedding dimension (default: 128)
        context_window: Context window size (default: 2)
        learning_rate: Hebbian learning rate (default: 0.01)
        meta_beta: Meta-plasticity factor (default: 0.999)
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        context_window: int = 2,
        learning_rate: float = 0.01,
        meta_beta: float = 0.999,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.learning_rate = learning_rate
        self.meta_beta = meta_beta
        
        # Word to index mapping
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.next_idx = 0
        
        # Embeddings (learnable)
        self.register_buffer(
            'embeddings',
            torch.randn(vocab_size, embedding_dim) * 0.1
        )
        
        # Normalize embeddings
        self.embeddings = F.normalize(self.embeddings, p=2, dim=1)
        
        # Usage tracking for meta-plasticity
        self.register_buffer(
            'usage_count',
            torch.zeros(vocab_size)
        )
    
    def add_word(self, word: str) -> int:
        """Add word to vocabulary, return index."""
        word = word.lower()
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        
        if self.next_idx >= self.vocab_size:
            # Vocabulary full, return special OOV index
            return -1
        
        idx = self.next_idx
        self.word_to_idx[word] = idx
        self.idx_to_word[idx] = word
        self.next_idx += 1
        return idx
    
    def build_vocab(self, corpus: List[str], min_count: int = 1):
        """
        Build vocabulary from corpus.
        
        Args:
            corpus: List of sentences
            min_count: Minimum word count to include
        """
        # Count word frequencies
        word_counts = defaultdict(int)
        for sentence in corpus:
            words = sentence.lower().split()
            for word in words:
                word_counts[word] += 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        
        # Add words to vocabulary
        for word, count in sorted_words:
            if count >= min_count and self.next_idx < self.vocab_size:
                self.add_word(word)
        
        print(f"Vocabulary size: {self.next_idx}")
    
    def learn_from_corpus(
        self,
        corpus: List[str],
        epochs: int = 1,
        verbose: bool = True
    ):
        """
        Learn embeddings from corpus via Hebbian co-occurrence.
        
        Skip-gram style: word embedding pulled toward context embeddings.
        
        Args:
            corpus: List of sentences
            epochs: Number of epochs
            verbose: Print progress
        """
        for epoch in range(epochs):
            total_updates = 0
            
            for sentence in corpus:
                words = sentence.lower().split()
                
                for i, word in enumerate(words):
                    word_idx = self.word_to_idx.get(word, -1)
                    if word_idx == -1:
                        continue
                    
                    # Get context words
                    context_start = max(0, i - self.context_window)
                    context_end = min(len(words), i + self.context_window + 1)
                    
                    for j in range(context_start, context_end):
                        if j == i:
                            continue
                        
                        ctx_word = words[j]
                        ctx_idx = self.word_to_idx.get(ctx_word, -1)
                        if ctx_idx == -1:
                            continue
                        
                        # Hebbian update: pull word toward context
                        self._hebbian_update(word_idx, ctx_idx)
                        total_updates += 1
            
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: {total_updates} updates")
    
    def _hebbian_update(self, word_idx: int, context_idx: int):
        """
        Hebbian update: strengthen connection between word and context.
        """
        # Get embeddings
        word_emb = self.embeddings[word_idx]
        ctx_emb = self.embeddings[context_idx]
        
        # Meta-plasticity
        effective_lr = self.learning_rate / (1.0 + self.meta_beta * self.usage_count[word_idx])
        
        # Hebbian: move word toward context
        delta = effective_lr * (ctx_emb - word_emb)
        self.embeddings[word_idx] = self.embeddings[word_idx] + delta
        
        # Normalize
        self.embeddings[word_idx] = F.normalize(self.embeddings[word_idx], p=2, dim=0)
        
        # Update usage
        self.usage_count[word_idx] = self.usage_count[word_idx] + 1
    
    def forward(
        self,
        text: Union[str, List[str]],
        pooling: str = 'mean'
    ) -> torch.Tensor:
        """
        Encode text to semantic embedding.
        
        Args:
            text: Input string or list of strings
            pooling: Pooling method ('mean', 'max', 'first')
        
        Returns:
            Embedding (embedding_dim,) or (batch, embedding_dim)
        """
        if isinstance(text, str):
            return self._encode_single(text, pooling)
        else:
            embeddings = [self._encode_single(t, pooling) for t in text]
            return torch.stack(embeddings)
    
    def _encode_single(self, text: str, pooling: str = 'mean') -> torch.Tensor:
        """Encode single text string."""
        words = text.lower().split()
        
        embeddings = []
        for word in words:
            idx = self.word_to_idx.get(word, -1)
            if idx != -1:
                embeddings.append(self.embeddings[idx])
        
        if len(embeddings) == 0:
            return torch.zeros(self.embedding_dim)
        
        emb_tensor = torch.stack(embeddings)
        
        if pooling == 'mean':
            pooled = emb_tensor.mean(dim=0)
        elif pooling == 'max':
            pooled = emb_tensor.max(dim=0)[0]
        elif pooling == 'first':
            pooled = emb_tensor[0]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        return F.normalize(pooled, p=2, dim=0)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.forward(text1)
        emb2 = self.forward(text2)
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    def most_similar(self, word: str, topk: int = 5) -> List[tuple]:
        """Find most similar words."""
        idx = self.word_to_idx.get(word.lower(), -1)
        if idx == -1:
            return []
        
        word_emb = self.embeddings[idx]
        
        # Compute similarities to all words
        sims = self.embeddings[:self.next_idx] @ word_emb
        
        # Get top-k (excluding the word itself)
        values, indices = torch.topk(sims, topk + 1)
        
        results = []
        for v, i in zip(values.tolist(), indices.tolist()):
            if i != idx:
                results.append((self.idx_to_word[i], v))
        
        return results[:topk]
    
    def get_effective_capacity(self, threshold: float = 1.0) -> float:
        """Compute effective capacity."""
        return (self.usage_count[:self.next_idx] > threshold).float().mean().item()
