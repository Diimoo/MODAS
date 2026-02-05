"""
Unit tests for Language Encoder module.
"""

import pytest
import torch
import torch.nn.functional as F

from modas.modules.language_encoder import LanguageEncoder, HebbianLanguageEncoder


class TestLanguageEncoder:
    """Tests for LanguageEncoder module."""
    
    @pytest.fixture
    def model(self):
        """Create language encoder for testing (without loading Word2Vec)."""
        return LanguageEncoder(output_dim=64, load_pretrained=False)
    
    def test_init(self, model):
        """Test model initialization."""
        assert model.output_dim == 64
        assert model.projection.shape == (64, 300)
    
    def test_forward_single_word(self, model):
        """Test encoding single word."""
        # Without pretrained model, uses random embeddings
        emb = model.forward("hello")
        
        assert emb.shape == (64,)
        # Should be unit-norm
        assert torch.allclose(emb.norm(), torch.tensor(1.0), atol=1e-5)
    
    def test_forward_multi_word(self, model):
        """Test encoding multi-word text."""
        emb = model.forward("hello world")
        
        assert emb.shape == (64,)
    
    def test_forward_batch(self, model):
        """Test encoding batch of texts."""
        texts = ["hello", "world", "test"]
        embs = model.forward(texts)
        
        assert embs.shape == (3, 64)
    
    def test_oov_handling(self, model):
        """Test out-of-vocabulary handling."""
        # Empty or OOV should return zero vector
        emb = model.forward("")
        assert emb.shape == (64,)


class TestHebbianLanguageEncoder:
    """Tests for HebbianLanguageEncoder module."""
    
    @pytest.fixture
    def model(self):
        """Create Hebbian language encoder."""
        return HebbianLanguageEncoder(vocab_size=100, embedding_dim=64)
    
    def test_init(self, model):
        """Test model initialization."""
        assert model.vocab_size == 100
        assert model.embedding_dim == 64
        assert model.embeddings.shape == (100, 64)
    
    def test_add_word(self, model):
        """Test adding words to vocabulary."""
        idx1 = model.add_word("hello")
        idx2 = model.add_word("world")
        idx3 = model.add_word("hello")  # Duplicate
        
        assert idx1 == 0
        assert idx2 == 1
        assert idx3 == 0  # Should return same index
    
    def test_build_vocab(self, model):
        """Test vocabulary building from corpus."""
        corpus = [
            "hello world",
            "hello there",
            "world peace",
        ]
        model.build_vocab(corpus)
        
        assert "hello" in model.word_to_idx
        assert "world" in model.word_to_idx
    
    def test_forward(self, model):
        """Test forward pass."""
        model.add_word("hello")
        model.add_word("world")
        
        emb = model.forward("hello world")
        
        assert emb.shape == (64,)
    
    def test_learn_from_corpus(self, model):
        """Test learning from corpus."""
        corpus = [
            "the cat sat",
            "the dog ran",
            "cat and dog",
        ]
        model.build_vocab(corpus)
        initial_embs = model.embeddings.clone()
        
        model.learn_from_corpus(corpus, epochs=1, verbose=False)
        
        # Embeddings should change
        assert not torch.allclose(model.embeddings, initial_embs)
    
    def test_similarity(self, model):
        """Test similarity computation."""
        model.add_word("hello")
        model.add_word("world")
        
        sim = model.similarity("hello", "world")
        
        assert isinstance(sim, float)
        assert -1 <= sim <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
