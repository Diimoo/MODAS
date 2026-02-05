"""
V1 Visual Cortex - Sparse Coding via Locally Competitive Algorithm (LCA)

Transforms raw images into sparse, discriminative features.
Based on Olshausen & Field (1996) with bio-plausible improvements.

Key features:
- LCA dynamics (lateral inhibition, recurrent processing)
- Meta-plasticity to prevent basis collapse (from MBM)
- Online Hebbian learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class V1SparseCoding(nn.Module):
    """
    V1 Visual Cortex sparse coding module.
    
    Uses Locally Competitive Algorithm (LCA) for inference
    and Hebbian learning with meta-plasticity for dictionary learning.
    
    Args:
        n_bases: Number of dictionary bases (default: 128)
        patch_size: Size of image patches (default: 16)
        n_channels: Number of color channels (default: 3 for RGB)
        lambda_sparse: Sparsity penalty (default: 0.1)
        tau: LCA time constant (default: 10.0)
        eta: Learning rate (default: 0.01)
        meta_beta: Meta-plasticity decay factor (default: 0.999)
        lca_iterations: Number of LCA iterations (default: 50)
    """
    
    def __init__(
        self,
        n_bases: int = 128,
        patch_size: int = 16,
        n_channels: int = 3,
        lambda_sparse: float = 0.5,
        tau: float = 10.0,
        eta: float = 0.01,
        meta_beta: float = 0.999,
        lca_iterations: int = 50,
    ):
        super().__init__()
        
        self.n_bases = n_bases
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.input_dim = patch_size * patch_size * n_channels
        
        # LCA parameters
        self.lambda_sparse = lambda_sparse
        self.tau = tau
        self.eta = eta
        self.lca_iterations = lca_iterations
        
        # Meta-plasticity (FROM MBM LESSON)
        self.meta_beta = meta_beta
        
        # Dictionary of basis functions (learnable)
        # Shape: (n_bases, input_dim)
        self.register_buffer(
            'dictionary',
            self._init_dictionary()
        )
        
        # Usage tracking for meta-plasticity
        self.register_buffer(
            'usage_count',
            torch.zeros(n_bases)
        )
        
        # Precompute Gram matrix for lateral inhibition
        self._update_gram_matrix()
    
    def _init_dictionary(self) -> torch.Tensor:
        """Initialize dictionary with random unit-norm bases."""
        dictionary = torch.randn(self.n_bases, self.input_dim)
        dictionary = F.normalize(dictionary, p=2, dim=1)
        return dictionary
    
    def _update_gram_matrix(self):
        """Update Gram matrix for efficient lateral inhibition."""
        # G = D @ D.T - I (subtract identity for self-inhibition)
        gram = self.dictionary @ self.dictionary.T
        gram = gram - torch.eye(self.n_bases, device=self.dictionary.device)
        self.register_buffer('gram_matrix', gram)
    
    def lca_inference(self, patch: torch.Tensor) -> torch.Tensor:
        """
        LCA inference via recurrent dynamics.
        
        Args:
            patch: Flattened image patch (input_dim,) or batch (batch, input_dim)
        
        Returns:
            Sparse code (n_bases,) or (batch, n_bases)
        """
        # Handle both single patch and batch
        if patch.dim() == 1:
            patch = patch.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = patch.shape[0]
        device = patch.device
        
        # Initialize membrane potentials
        u = torch.zeros(batch_size, self.n_bases, device=device)
        
        # Feed-forward drive: b = D @ x
        b = patch @ self.dictionary.T  # (batch, n_bases)
        
        # Recurrent dynamics with stable time step
        dt = 0.1  # Small time step for stability
        for _ in range(self.lca_iterations):
            # Soft threshold to get activities
            a = F.softshrink(u, lambd=self.lambda_sparse)
            
            # Lateral inhibition
            inhibition = a @ self.gram_matrix  # (batch, n_bases)
            
            # Update membrane potential (stable integration)
            du = (b - u - inhibition) / self.tau
            u = u + dt * du
        
        # Final activities
        activities = F.softshrink(u, lambd=self.lambda_sparse)
        
        if squeeze_output:
            activities = activities.squeeze(0)
        
        return activities
    
    def forward(
        self,
        image: torch.Tensor,
        stride: int = 8,
        pool: str = 'max'
    ) -> torch.Tensor:
        """
        Process full image through V1.
        
        Args:
            image: Input image (C, H, W) or (batch, C, H, W)
            stride: Patch extraction stride (default: 8)
            pool: Pooling method across patches ('max', 'mean', 'concat')
        
        Returns:
            Sparse code (n_bases,) or (batch, n_bases) if pool='max'/'mean'
            Or (batch, n_patches, n_bases) if pool='concat'
        """
        # Handle single image vs batch
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = image.shape[0]
        device = image.device
        
        # Move dictionary to same device
        if self.dictionary.device != device:
            self.dictionary = self.dictionary.to(device)
            self._update_gram_matrix()
        
        # Extract patches
        patches = self._extract_patches(image, stride)  # (batch, n_patches, input_dim)
        n_patches = patches.shape[1]
        
        # Process all patches
        patches_flat = patches.reshape(-1, self.input_dim)  # (batch*n_patches, input_dim)
        codes_flat = self.lca_inference(patches_flat)  # (batch*n_patches, n_bases)
        codes = codes_flat.reshape(batch_size, n_patches, self.n_bases)
        
        # Pool across patches
        if pool == 'max':
            output = codes.max(dim=1)[0]  # (batch, n_bases)
        elif pool == 'mean':
            output = codes.mean(dim=1)  # (batch, n_bases)
        elif pool == 'concat':
            output = codes  # (batch, n_patches, n_bases)
        else:
            raise ValueError(f"Unknown pooling method: {pool}")
        
        if squeeze_output and pool != 'concat':
            output = output.squeeze(0)
        
        return output
    
    def _extract_patches(
        self,
        image: torch.Tensor,
        stride: int
    ) -> torch.Tensor:
        """
        Extract overlapping patches from image.
        
        Args:
            image: (batch, C, H, W)
            stride: Patch stride
        
        Returns:
            patches: (batch, n_patches, patch_size*patch_size*C)
        """
        batch_size, C, H, W = image.shape
        
        # Use unfold to extract patches
        patches = image.unfold(2, self.patch_size, stride).unfold(3, self.patch_size, stride)
        # Shape: (batch, C, n_h, n_w, patch_size, patch_size)
        
        n_h, n_w = patches.shape[2], patches.shape[3]
        
        # Reshape to (batch, n_patches, C*patch_size*patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.reshape(batch_size, n_h * n_w, -1)
        
        # Normalize patches (zero mean, unit variance)
        patches = patches - patches.mean(dim=-1, keepdim=True)
        patches_std = patches.std(dim=-1, keepdim=True)
        patches = patches / (patches_std + 1e-8)
        
        return patches
    
    def learn(self, patch: torch.Tensor, code: torch.Tensor) -> float:
        """
        Update dictionary via Hebbian learning with meta-plasticity.
        
        Args:
            patch: Flattened image patch (input_dim,)
            code: Sparse code from LCA inference (n_bases,)
        
        Returns:
            Reconstruction error (MSE)
        """
        # Ensure 1D
        if patch.dim() > 1:
            patch = patch.flatten()
        if code.dim() > 1:
            code = code.flatten()
        
        # Reconstruction error
        reconstruction = code @ self.dictionary  # (input_dim,)
        error = patch - reconstruction
        mse = (error ** 2).mean().item()
        
        # Hebbian update (outer product)
        # delta_D = eta * a * error^T
        delta = torch.outer(code, error)  # (n_bases, input_dim)
        
        # Meta-plasticity: reduce LR for overused bases (FROM MBM)
        effective_lr = self.eta / (1.0 + self.meta_beta * self.usage_count)
        
        # Update dictionary
        self.dictionary = self.dictionary + effective_lr.unsqueeze(1) * delta
        
        # Re-normalize to unit norm
        self.dictionary = F.normalize(self.dictionary, p=2, dim=1)
        
        # Update usage count
        self.usage_count = self.usage_count + code.abs()
        
        # Update Gram matrix
        self._update_gram_matrix()
        
        return mse
    
    def learn_batch(
        self,
        patches: torch.Tensor,
        codes: torch.Tensor
    ) -> float:
        """
        Batch learning update.
        
        Args:
            patches: (batch, input_dim)
            codes: (batch, n_bases)
        
        Returns:
            Mean reconstruction error
        """
        # Reconstruction error
        reconstruction = codes @ self.dictionary  # (batch, input_dim)
        error = patches - reconstruction
        mse = (error ** 2).mean().item()
        
        # Batch Hebbian update
        delta = codes.T @ error / codes.shape[0]  # (n_bases, input_dim)
        
        # Meta-plasticity
        effective_lr = self.eta / (1.0 + self.meta_beta * self.usage_count)
        
        # Update
        self.dictionary = self.dictionary + effective_lr.unsqueeze(1) * delta
        self.dictionary = F.normalize(self.dictionary, p=2, dim=1)
        
        # Update usage
        self.usage_count = self.usage_count + codes.abs().mean(dim=0)
        
        # Update Gram matrix
        self._update_gram_matrix()
        
        return mse
    
    def get_sparsity(self, codes: torch.Tensor, threshold: float = 0.01) -> float:
        """Compute activation sparsity (fraction of INACTIVE units). Higher = sparser."""
        return (codes.abs() <= threshold).float().mean().item()
    
    def get_effective_capacity(self, threshold: float = 0.01) -> float:
        """Compute effective capacity (fraction of used bases)."""
        return (self.usage_count > threshold).float().mean().item()
    
    def visualize_dictionary(self) -> torch.Tensor:
        """
        Reshape dictionary for visualization.
        
        Returns:
            filters: (n_bases, patch_size, patch_size, n_channels)
        """
        filters = self.dictionary.reshape(
            self.n_bases, self.n_channels, self.patch_size, self.patch_size
        )
        filters = filters.permute(0, 2, 3, 1)  # (n_bases, H, W, C)
        return filters
    
    def reset_usage(self):
        """Reset usage counts (call at start of new training phase)."""
        self.usage_count.zero_()
    
    def state_dict_custom(self) -> dict:
        """Get custom state dict including buffers."""
        return {
            'dictionary': self.dictionary.clone(),
            'usage_count': self.usage_count.clone(),
            'gram_matrix': self.gram_matrix.clone(),
        }
    
    def load_state_dict_custom(self, state_dict: dict):
        """Load custom state dict."""
        self.dictionary = state_dict['dictionary']
        self.usage_count = state_dict['usage_count']
        self.gram_matrix = state_dict['gram_matrix']


def extract_patches(
    image: torch.Tensor,
    size: int = 16,
    stride: int = 8
) -> torch.Tensor:
    """
    Utility function to extract patches from an image.
    
    Args:
        image: (C, H, W) or (H, W) tensor
        size: Patch size
        stride: Patch stride
    
    Returns:
        patches: (n_patches, size*size*C) flattened patches
    """
    if image.dim() == 2:
        image = image.unsqueeze(0)  # Add channel dim
    
    C, H, W = image.shape
    
    # Extract patches using unfold
    patches = image.unfold(1, size, stride).unfold(2, size, stride)
    # Shape: (C, n_h, n_w, size, size)
    
    n_h, n_w = patches.shape[1], patches.shape[2]
    
    # Reshape to (n_patches, C*size*size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    patches = patches.reshape(n_h * n_w, -1)
    
    return patches
