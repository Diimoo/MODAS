"""
A1 Auditory Cortex - Temporal Sparse Coding

Transforms audio into sparse temporal features.
1D variant of V1 sparse coding applied to spectrograms.

Key features:
- LCA dynamics for inference
- Temporal basis functions (frequency-selective, time-localized)
- Meta-plasticity to prevent basis collapse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class A1SparseCoding(nn.Module):
    """
    A1 Auditory Cortex sparse coding module.
    
    Uses LCA for inference on audio spectrograms and Hebbian learning
    with meta-plasticity for dictionary learning.
    
    Args:
        n_bases: Number of dictionary bases (default: 128)
        segment_length: Audio segment length in samples (default: 3200 = 200ms @ 16kHz)
        sample_rate: Audio sample rate (default: 16000)
        n_fft: FFT size for spectrogram (default: 512)
        hop_length: Hop length for spectrogram (default: 160)
        lambda_sparse: Sparsity penalty (default: 0.1)
        tau: LCA time constant (default: 10.0)
        eta: Learning rate (default: 0.01)
        meta_beta: Meta-plasticity decay factor (default: 0.999)
        lca_iterations: Number of LCA iterations (default: 50)
    """
    
    def __init__(
        self,
        n_bases: int = 128,
        segment_length: int = 3200,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        lambda_sparse: float = 0.5,
        tau: float = 10.0,
        eta: float = 0.01,
        meta_beta: float = 0.999,
        lca_iterations: int = 50,
    ):
        super().__init__()
        
        self.n_bases = n_bases
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # LCA parameters
        self.lambda_sparse = lambda_sparse
        self.tau = tau
        self.eta = eta
        self.lca_iterations = lca_iterations
        
        # Meta-plasticity (FROM MBM LESSON)
        self.meta_beta = meta_beta
        
        # Compute spectrogram dimensions
        n_freq_bins = n_fft // 2 + 1
        n_time_frames = (segment_length - n_fft) // hop_length + 1
        self.spec_dim = n_freq_bins * n_time_frames
        
        # Dictionary of temporal basis functions (learnable)
        self.register_buffer(
            'dictionary',
            self._init_dictionary()
        )
        
        # Usage tracking for meta-plasticity
        self.register_buffer(
            'usage_count',
            torch.zeros(n_bases)
        )
        
        # Precompute Gram matrix
        self._update_gram_matrix()
        
        # Store spectrogram stats for normalization
        self.register_buffer('spec_mean', torch.zeros(1))
        self.register_buffer('spec_std', torch.ones(1))
        self.stats_initialized = False
    
    def _init_dictionary(self) -> torch.Tensor:
        """Initialize dictionary with random unit-norm bases."""
        dictionary = torch.randn(self.n_bases, self.spec_dim)
        dictionary = F.normalize(dictionary, p=2, dim=1)
        return dictionary
    
    def _update_gram_matrix(self):
        """Update Gram matrix for efficient lateral inhibition."""
        gram = self.dictionary @ self.dictionary.T
        gram = gram - torch.eye(self.n_bases, device=self.dictionary.device)
        self.register_buffer('gram_matrix', gram)
    
    def compute_spectrogram(
        self,
        audio: torch.Tensor,
        return_complex: bool = False
    ) -> torch.Tensor:
        """
        Compute spectrogram from audio waveform.
        
        Args:
            audio: Audio waveform (segment_length,) or (batch, segment_length)
            return_complex: Return complex spectrogram (default: False, returns magnitude)
        
        Returns:
            Spectrogram (n_freq, n_time) or (batch, n_freq, n_time)
        """
        # Handle single audio vs batch
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Compute STFT
        window = torch.hann_window(self.n_fft, device=audio.device)
        
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
            center=False
        )
        
        if return_complex:
            output = spec
        else:
            # Magnitude spectrogram
            output = spec.abs()
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output
    
    def lca_inference(self, spec_flat: torch.Tensor) -> torch.Tensor:
        """
        LCA inference via recurrent dynamics.
        
        Args:
            spec_flat: Flattened spectrogram (spec_dim,) or (batch, spec_dim)
        
        Returns:
            Sparse code (n_bases,) or (batch, n_bases)
        """
        # Handle both single input and batch
        if spec_flat.dim() == 1:
            spec_flat = spec_flat.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = spec_flat.shape[0]
        device = spec_flat.device
        
        # Initialize membrane potentials
        u = torch.zeros(batch_size, self.n_bases, device=device)
        
        # Feed-forward drive
        b = spec_flat @ self.dictionary.T  # (batch, n_bases)
        
        # Recurrent dynamics
        for _ in range(self.lca_iterations):
            # Soft threshold
            a = F.softshrink(u, lambd=self.lambda_sparse)
            
            # Lateral inhibition
            inhibition = a @ self.gram_matrix
            
            # Update
            du = (b - u - inhibition) / self.tau
            u = u + du
        
        # Final activities
        activities = F.softshrink(u, lambd=self.lambda_sparse)
        
        if squeeze_output:
            activities = activities.squeeze(0)
        
        return activities
    
    def forward(
        self,
        audio: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Process audio segment through A1.
        
        Args:
            audio: Audio waveform (segment_length,) or (batch, segment_length)
            normalize: Normalize spectrogram (default: True)
        
        Returns:
            Sparse code (n_bases,) or (batch, n_bases)
        """
        # Handle single audio vs batch
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        device = audio.device
        
        # Move dictionary to same device
        if self.dictionary.device != device:
            self.dictionary = self.dictionary.to(device)
            self._update_gram_matrix()
        
        # Compute spectrogram
        spec = self.compute_spectrogram(audio)  # (batch, n_freq, n_time)
        
        # Flatten spectrogram
        spec_flat = spec.reshape(spec.shape[0], -1)  # (batch, spec_dim)
        
        # Normalize
        if normalize:
            if self.stats_initialized:
                spec_flat = (spec_flat - self.spec_mean) / (self.spec_std + 1e-8)
            else:
                # Online normalization
                spec_flat = spec_flat - spec_flat.mean(dim=-1, keepdim=True)
                spec_flat = spec_flat / (spec_flat.std(dim=-1, keepdim=True) + 1e-8)
        
        # LCA inference
        code = self.lca_inference(spec_flat)
        
        if squeeze_output:
            code = code.squeeze(0)
        
        return code
    
    def learn(self, spec_flat: torch.Tensor, code: torch.Tensor) -> float:
        """
        Update dictionary via Hebbian learning with meta-plasticity.
        
        Args:
            spec_flat: Flattened spectrogram (spec_dim,)
            code: Sparse code from LCA inference (n_bases,)
        
        Returns:
            Reconstruction error (MSE)
        """
        # Ensure 1D
        if spec_flat.dim() > 1:
            spec_flat = spec_flat.flatten()
        if code.dim() > 1:
            code = code.flatten()
        
        # Reconstruction error
        reconstruction = code @ self.dictionary
        error = spec_flat - reconstruction
        mse = (error ** 2).mean().item()
        
        # Hebbian update
        delta = torch.outer(code, error)
        
        # Meta-plasticity
        effective_lr = self.eta / (1.0 + self.meta_beta * self.usage_count)
        
        # Update dictionary
        self.dictionary = self.dictionary + effective_lr.unsqueeze(1) * delta
        self.dictionary = F.normalize(self.dictionary, p=2, dim=1)
        
        # Update usage
        self.usage_count = self.usage_count + code.abs()
        
        # Update Gram matrix
        self._update_gram_matrix()
        
        return mse
    
    def learn_batch(
        self,
        spec_flat: torch.Tensor,
        codes: torch.Tensor
    ) -> float:
        """
        Batch learning update.
        
        Args:
            spec_flat: (batch, spec_dim)
            codes: (batch, n_bases)
        
        Returns:
            Mean reconstruction error
        """
        # Reconstruction error
        reconstruction = codes @ self.dictionary
        error = spec_flat - reconstruction
        mse = (error ** 2).mean().item()
        
        # Batch Hebbian update
        delta = codes.T @ error / codes.shape[0]
        
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
    
    def update_stats(self, spec_flat: torch.Tensor, momentum: float = 0.99):
        """Update running statistics for normalization."""
        batch_mean = spec_flat.mean()
        batch_std = spec_flat.std()
        
        if not self.stats_initialized:
            self.spec_mean = batch_mean
            self.spec_std = batch_std
            self.stats_initialized = True
        else:
            self.spec_mean = momentum * self.spec_mean + (1 - momentum) * batch_mean
            self.spec_std = momentum * self.spec_std + (1 - momentum) * batch_std
    
    def get_sparsity(self, codes: torch.Tensor, threshold: float = 0.01) -> float:
        """Compute activation sparsity (fraction of INACTIVE units). Higher = sparser."""
        return (codes.abs() <= threshold).float().mean().item()
    
    def get_effective_capacity(self, threshold: float = 0.01) -> float:
        """Compute effective capacity."""
        return (self.usage_count > threshold).float().mean().item()
    
    def visualize_dictionary(self) -> torch.Tensor:
        """
        Reshape dictionary for visualization as spectrograms.
        
        Returns:
            filters: (n_bases, n_freq, n_time)
        """
        n_freq = self.n_fft // 2 + 1
        n_time = (self.segment_length - self.n_fft) // self.hop_length + 1
        
        filters = self.dictionary.reshape(self.n_bases, n_freq, n_time)
        return filters
    
    def reset_usage(self):
        """Reset usage counts."""
        self.usage_count.zero_()


def segment_audio(
    waveform: torch.Tensor,
    length: int = 3200,
    stride: int = 1600
) -> torch.Tensor:
    """
    Segment audio waveform into overlapping segments.
    
    Args:
        waveform: Audio waveform (n_samples,) or (channels, n_samples)
        length: Segment length in samples
        stride: Segment stride in samples
    
    Returns:
        segments: (n_segments, length)
    """
    if waveform.dim() == 2:
        # Take first channel if stereo
        waveform = waveform[0]
    
    n_samples = waveform.shape[0]
    
    # Calculate number of segments
    n_segments = (n_samples - length) // stride + 1
    
    if n_segments <= 0:
        # Pad if audio is too short
        pad_length = length - n_samples
        waveform = F.pad(waveform, (0, pad_length))
        return waveform.unsqueeze(0)
    
    # Extract segments
    segments = []
    for i in range(n_segments):
        start = i * stride
        end = start + length
        segments.append(waveform[start:end])
    
    return torch.stack(segments)
