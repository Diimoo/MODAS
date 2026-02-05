"""
Unit tests for A1 Sparse Coding module.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from modas.modules.a1_sparse_coding import A1SparseCoding, segment_audio


class TestA1SparseCoding:
    """Tests for A1SparseCoding module."""
    
    @pytest.fixture
    def model(self):
        """Create A1 model for testing."""
        return A1SparseCoding(
            n_bases=64,
            segment_length=1600,
            sample_rate=16000,
            n_fft=256,
            hop_length=80,
            lca_iterations=20,
        )
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio segment."""
        return torch.randn(1600)
    
    def test_init(self, model):
        """Test model initialization."""
        assert model.n_bases == 64
        assert model.segment_length == 1600
        assert model.sample_rate == 16000
        
        # Dictionary should be unit-norm
        norms = model.dictionary.norm(dim=1)
        assert torch.allclose(norms, torch.ones(64), atol=1e-5)
    
    def test_compute_spectrogram(self, model, sample_audio):
        """Test spectrogram computation."""
        spec = model.compute_spectrogram(sample_audio)
        
        assert spec.dim() == 2  # (n_freq, n_time)
        assert spec.shape[0] == model.n_fft // 2 + 1
    
    def test_forward_single(self, model, sample_audio):
        """Test forward pass on single audio."""
        code = model.forward(sample_audio)
        
        assert code.shape == (64,)
    
    def test_forward_batch(self, model):
        """Test forward pass on batch."""
        batch = torch.randn(8, 1600)
        codes = model.forward(batch)
        
        assert codes.shape == (8, 64)
    
    def test_learn(self, model, sample_audio):
        """Test learning update."""
        spec = model.compute_spectrogram(sample_audio)
        spec_flat = spec.flatten()
        spec_flat = (spec_flat - spec_flat.mean()) / (spec_flat.std() + 1e-8)
        
        code = model.lca_inference(spec_flat)
        initial_dict = model.dictionary.clone()
        
        mse = model.learn(spec_flat, code)
        
        assert isinstance(mse, float)
        assert mse >= 0
        
        # Dictionary should change
        assert not torch.allclose(model.dictionary, initial_dict)
        
        # Dictionary should remain unit-norm
        norms = model.dictionary.norm(dim=1)
        assert torch.allclose(norms, torch.ones(64), atol=1e-5)
    
    def test_sparsity(self, model, sample_audio):
        """Test sparsity computation."""
        code = model.forward(sample_audio)
        sparsity = model.get_sparsity(code)
        
        assert 0 <= sparsity <= 1
    
    def test_visualize_dictionary(self, model):
        """Test dictionary visualization."""
        filters = model.visualize_dictionary()
        
        n_freq = model.n_fft // 2 + 1
        assert filters.shape[0] == 64
        assert filters.shape[1] == n_freq


class TestSegmentAudio:
    """Tests for audio segmentation utility."""
    
    def test_segment_audio_basic(self):
        """Test basic audio segmentation."""
        waveform = torch.randn(16000)  # 1 second at 16kHz
        segments = segment_audio(waveform, length=3200, stride=1600)
        
        assert segments.shape[1] == 3200
        assert segments.shape[0] >= 1
    
    def test_segment_audio_short(self):
        """Test segmentation of short audio."""
        waveform = torch.randn(1000)  # Shorter than segment length
        segments = segment_audio(waveform, length=3200, stride=1600)
        
        assert segments.shape == (1, 3200)
    
    def test_segment_audio_stereo(self):
        """Test segmentation of stereo audio."""
        waveform = torch.randn(2, 16000)  # Stereo
        segments = segment_audio(waveform, length=3200, stride=1600)
        
        assert segments.shape[1] == 3200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
