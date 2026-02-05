"""
Unit tests for V1 Sparse Coding module.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from modas.modules.v1_sparse_coding import V1SparseCoding, extract_patches


class TestV1SparseCoding:
    """Tests for V1SparseCoding module."""
    
    @pytest.fixture
    def model(self):
        """Create V1 model for testing."""
        return V1SparseCoding(
            n_bases=64,
            patch_size=8,
            n_channels=3,
            lca_iterations=20,
        )
    
    @pytest.fixture
    def sample_patch(self):
        """Create sample patch."""
        patch = torch.randn(8 * 8 * 3)
        patch = patch - patch.mean()
        patch = patch / patch.std()
        return patch
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        return torch.randn(3, 64, 64)
    
    def test_init(self, model):
        """Test model initialization."""
        assert model.n_bases == 64
        assert model.patch_size == 8
        assert model.dictionary.shape == (64, 8 * 8 * 3)
        
        # Dictionary should be unit-norm
        norms = model.dictionary.norm(dim=1)
        assert torch.allclose(norms, torch.ones(64), atol=1e-5)
    
    def test_lca_inference_single(self, model, sample_patch):
        """Test LCA inference on single patch."""
        code = model.lca_inference(sample_patch)
        
        assert code.shape == (64,)
        # Code should be sparse
        sparsity = (code.abs() < 0.01).float().mean()
        assert sparsity > 0.5  # At least 50% sparse
    
    def test_lca_inference_batch(self, model):
        """Test LCA inference on batch."""
        batch = torch.randn(16, 8 * 8 * 3)
        codes = model.lca_inference(batch)
        
        assert codes.shape == (16, 64)
    
    def test_forward_single_image(self, model, sample_image):
        """Test forward pass on single image."""
        code = model.forward(sample_image)
        
        assert code.shape == (64,)
    
    def test_forward_batch(self, model):
        """Test forward pass on batch of images."""
        batch = torch.randn(4, 3, 64, 64)
        codes = model.forward(batch)
        
        assert codes.shape == (4, 64)
    
    def test_learn(self, model, sample_patch):
        """Test learning update."""
        code = model.lca_inference(sample_patch)
        initial_dict = model.dictionary.clone()
        
        mse = model.learn(sample_patch, code)
        
        assert isinstance(mse, float)
        assert mse >= 0
        
        # Dictionary should change
        assert not torch.allclose(model.dictionary, initial_dict)
        
        # Dictionary should remain unit-norm
        norms = model.dictionary.norm(dim=1)
        assert torch.allclose(norms, torch.ones(64), atol=1e-5)
    
    def test_learn_batch(self, model):
        """Test batch learning."""
        patches = torch.randn(32, 8 * 8 * 3)
        codes = model.lca_inference(patches)
        
        mse = model.learn_batch(patches, codes)
        
        assert isinstance(mse, float)
    
    def test_usage_tracking(self, model, sample_patch):
        """Test usage count tracking."""
        initial_usage = model.usage_count.clone()
        
        code = model.lca_inference(sample_patch)
        model.learn(sample_patch, code)
        
        # Usage should increase for active units
        assert (model.usage_count >= initial_usage).all()
    
    def test_meta_plasticity(self, model):
        """Test meta-plasticity reduces learning rate for overused bases."""
        # Simulate heavy usage of first basis
        model.usage_count[0] = 1000
        model.usage_count[1] = 0
        
        patch = torch.randn(8 * 8 * 3)
        code = torch.zeros(64)
        code[0] = 1.0  # Activate first basis
        code[1] = 1.0  # Activate second basis
        
        initial_dict = model.dictionary.clone()
        model.learn(patch, code)
        
        # Change in heavily used basis should be smaller
        change_0 = (model.dictionary[0] - initial_dict[0]).abs().sum()
        change_1 = (model.dictionary[1] - initial_dict[1]).abs().sum()
        
        # Basis 1 should change more than basis 0 (less used)
        assert change_1 > change_0
    
    def test_sparsity_metric(self, model, sample_patch):
        """Test sparsity computation."""
        code = model.lca_inference(sample_patch)
        sparsity = model.get_sparsity(code)
        
        assert 0 <= sparsity <= 1
    
    def test_visualize_dictionary(self, model):
        """Test dictionary visualization."""
        filters = model.visualize_dictionary()
        
        assert filters.shape == (64, 8, 8, 3)
    
    def test_reset_usage(self, model, sample_patch):
        """Test usage reset."""
        code = model.lca_inference(sample_patch)
        model.learn(sample_patch, code)
        
        assert model.usage_count.sum() > 0
        
        model.reset_usage()
        
        assert model.usage_count.sum() == 0


class TestExtractPatches:
    """Tests for patch extraction utility."""
    
    def test_extract_patches_grayscale(self):
        """Test patch extraction from grayscale image."""
        image = torch.randn(1, 32, 32)
        patches = extract_patches(image, size=8, stride=8)
        
        assert patches.shape == (16, 8 * 8)  # 4x4 grid, flattened
    
    def test_extract_patches_rgb(self):
        """Test patch extraction from RGB image."""
        image = torch.randn(3, 32, 32)
        patches = extract_patches(image, size=8, stride=8)
        
        assert patches.shape == (16, 8 * 8 * 3)
    
    def test_extract_patches_overlap(self):
        """Test overlapping patch extraction."""
        image = torch.randn(3, 32, 32)
        patches = extract_patches(image, size=8, stride=4)
        
        # Should have more patches with smaller stride
        assert patches.shape[0] > 16


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
