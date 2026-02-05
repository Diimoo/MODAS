"""
ATL Semantic Hub Training

Train ATL binding using Contrastive Hebbian Learning.
This is THE critical test of MODAS.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm

from modas.modules.atl_semantic_hub import ATLSemanticHub
from modas.modules.v1_sparse_coding import V1SparseCoding
from modas.modules.language_encoder import LanguageEncoder
from modas.modules.a1_sparse_coding import A1SparseCoding
from modas.utils.checkpointing import CheckpointManager
from modas.utils.visualization import plot_discrimination_results, plot_training_curves


class ATLTrainer:
    """
    Trainer for ATL semantic hub.
    
    Args:
        atl: ATL model
        v1: Pretrained V1 model
        language: Language encoder
        a1: Optional pretrained A1 model
        device: Training device
        checkpoint_dir: Checkpoint directory
    """
    
    def __init__(
        self,
        atl: ATLSemanticHub,
        v1: V1SparseCoding,
        language: LanguageEncoder,
        a1: Optional[A1SparseCoding] = None,
        device: str = 'cpu',
        checkpoint_dir: str = 'experiments/phase4_atl_binding/checkpoints',
        log_interval: int = 100,
        val_interval: int = 10,
    ):
        self.atl = atl.to(device)
        self.v1 = v1.to(device)
        self.language = language
        self.a1 = a1.to(device) if a1 is not None else None
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.val_interval = val_interval
        
        # Freeze pretrained modules
        for param in self.v1.parameters():
            param.requires_grad = False
        if self.a1 is not None:
            for param in self.a1.parameters():
                param.requires_grad = False
        
        self.history = {
            'similarity': [],
            'margin': [],
            'discrimination': [],
            'capacity': [],
        }
        
        self.ckpt_manager = CheckpointManager(
            str(self.checkpoint_dir),
            max_checkpoints=5,
            metric_name='discrimination',
            higher_is_better=True,
        )
    
    def get_features(
        self,
        image: torch.Tensor,
        text: str,
        audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Extract features from all modalities."""
        # Visual features
        with torch.no_grad():
            vis_code = self.v1.forward(image)
            if vis_code.dim() > 1:
                vis_code = vis_code.squeeze(0)
        
        # Language features
        lang_emb = self.language.forward(text)
        if lang_emb.device != self.device:
            lang_emb = lang_emb.to(self.device)
        
        # Audio features (optional)
        aud_code = None
        if audio is not None and self.a1 is not None:
            with torch.no_grad():
                aud_code = self.a1.forward(audio)
                if aud_code.dim() > 1:
                    aud_code = aud_code.squeeze(0)
        
        return vis_code, lang_emb, aud_code
    
    def train_epoch(
        self,
        dataloader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        total_sim = 0
        total_margin = 0
        n_samples = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            texts = batch['texts']
            audios = batch.get('audios')
            if audios is not None:
                audios = audios.to(self.device)
            
            batch_sim = 0
            batch_margin = 0
            
            for i in range(len(images)):
                audio = audios[i] if audios is not None else None
                
                # Get features
                vis_code, lang_emb, aud_code = self.get_features(
                    images[i], texts[i], audio
                )
                
                # Bind
                sim, margin = self.atl.bind(vis_code, lang_emb, aud_code)
                
                batch_sim += sim.item()
                batch_margin += margin.item()
            
            batch_sim /= len(images)
            batch_margin /= len(images)
            
            total_sim += batch_sim * len(images)
            total_margin += batch_margin * len(images)
            n_samples += len(images)
            
            if (batch_idx + 1) % self.log_interval == 0:
                pbar.set_postfix({
                    'Sim': f'{batch_sim:.3f}',
                    'Margin': f'{batch_margin:.3f}',
                })
        
        avg_sim = total_sim / n_samples
        avg_margin = total_margin / n_samples
        
        self.history['similarity'].append(avg_sim)
        self.history['margin'].append(avg_margin)
        
        return {'similarity': avg_sim, 'margin': avg_margin}
    
    def validate(
        self,
        test_data: List[Tuple[torch.Tensor, str]],
    ) -> Dict[str, float]:
        """
        Validate ATL binding.
        
        THE CRITICAL TEST: Compute discrimination score.
        """
        matched_sims = []
        mismatched_sims = []
        
        # Get features for all samples
        features = []
        for image, text in test_data:
            image = image.to(self.device)
            vis_code, lang_emb, _ = self.get_features(image, text)
            features.append((vis_code, lang_emb, text))
        
        # Compute matched similarities
        for vis_code, lang_emb, _ in features:
            sim = self.atl.compute_cross_modal_similarity(vis_code, lang_emb)
            matched_sims.append(sim.item())
        
        # Compute mismatched similarities
        for i, (vis_code, _, _) in enumerate(features):
            for j, (_, lang_emb, _) in enumerate(features):
                if i != j:
                    sim = self.atl.compute_cross_modal_similarity(vis_code, lang_emb)
                    mismatched_sims.append(sim.item())
        
        # Discrimination score
        disc = np.mean(matched_sims) - np.mean(mismatched_sims)
        
        # Capacity
        capacity = self.atl.get_effective_capacity()
        
        self.history['discrimination'].append(disc)
        self.history['capacity'].append(capacity)
        
        return {
            'discrimination': disc,
            'mean_matched': np.mean(matched_sims),
            'mean_mismatched': np.mean(mismatched_sims),
            'capacity': capacity,
            'matched_sims': matched_sims,
            'mismatched_sims': mismatched_sims,
        }
    
    def check_convergence(self, metrics: Dict[str, float]) -> Tuple[str, str]:
        """
        Check ATL binding quality.
        
        Thresholds (from spec):
        - EXCELLENT: > 0.2
        - GOOD: 0.15 - 0.2
        - MARGINAL: 0.1 - 0.15
        - FAIL: < 0.1
        """
        disc = metrics.get('discrimination', 0)
        
        if disc > 0.2:
            return 'EXCELLENT', '✓ EXCELLENT (>0.2)'
        elif disc > 0.15:
            return 'GOOD', '✓ GOOD (0.15-0.2)'
        elif disc > 0.1:
            return 'MARGINAL', '⚠ MARGINAL (0.1-0.15)'
        else:
            return 'FAIL', '✗ FAIL (<0.1)'
    
    def train(
        self,
        train_dataloader,
        test_data: List[Tuple[torch.Tensor, str]],
        n_epochs: int = 200,
        early_stop_threshold: float = 0.15,
    ) -> Dict[str, float]:
        """
        Full ATL training loop.
        
        Args:
            train_dataloader: Training data
            test_data: Test samples for validation
            n_epochs: Maximum epochs
            early_stop_threshold: Stop if discrimination exceeds this
        
        Returns:
            Final metrics
        """
        print(f"Starting ATL binding training for {n_epochs} epochs")
        print(f"Success threshold: discrimination > {early_stop_threshold}")
        print("=" * 50)
        
        best_metrics = None
        best_disc = -float('inf')
        
        for epoch in range(1, n_epochs + 1):
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            if epoch % self.val_interval == 0 or epoch == 1:
                val_metrics = self.validate(test_data)
                
                status, status_str = self.check_convergence(val_metrics)
                
                print(f"\nEpoch {epoch} Validation:")
                print(f"  Similarity: {train_metrics['similarity']:.3f}")
                print(f"  Margin: {train_metrics['margin']:.3f}")
                print(f"  Discrimination: {val_metrics['discrimination']:.3f}")
                print(f"  Matched: {val_metrics['mean_matched']:.3f}")
                print(f"  Mismatched: {val_metrics['mean_mismatched']:.3f}")
                print(f"  Capacity: {val_metrics['capacity']:.2%}")
                print(f"  Status: {status_str}")
                
                # Save checkpoint
                self.ckpt_manager.save(
                    self.atl, None, epoch,
                    {**train_metrics, **{k: v for k, v in val_metrics.items() 
                                        if not isinstance(v, list)}}
                )
                
                if val_metrics['discrimination'] > best_disc:
                    best_disc = val_metrics['discrimination']
                    best_metrics = val_metrics
                
                # Early stop on success
                if status in ['EXCELLENT', 'GOOD']:
                    print(f"\n{'='*50}")
                    print(f"ATL BINDING {status} at epoch {epoch}!")
                    print(f"Discrimination: {val_metrics['discrimination']:.3f}")
                    print(f"{'='*50}")
                    break
        
        # Save final model
        final_path = self.checkpoint_dir / 'atl_final.pt'
        torch.save({
            'model_state_dict': self.atl.state_dict(),
            'history': self.history,
            'metrics': best_metrics,
        }, final_path)
        print(f"Final model saved: {final_path}")
        
        return best_metrics or {}
    
    def visualize(self, save_dir: Optional[str] = None):
        """Visualize results."""
        if save_dir is None:
            save_dir = str(self.checkpoint_dir / 'figures')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Training curves
        if len(self.history['similarity']) > 0:
            plot_training_curves(
                {k: v for k, v in self.history.items() 
                 if k not in ['matched_sims', 'mismatched_sims']},
                title='ATL Training Progress',
                save_path=f'{save_dir}/atl_training.png',
                show=False,
            )


def train_atl(
    v1_path: str,
    data_source,
    n_epochs: int = 200,
    n_prototypes: int = 200,
    feature_dim: int = 128,
    device: str = 'cpu',
    checkpoint_dir: str = 'experiments/phase4_atl_binding/checkpoints',
    a1_path: Optional[str] = None,
    **kwargs
) -> ATLSemanticHub:
    """
    Train ATL semantic hub.
    
    Args:
        v1_path: Path to pretrained V1 model
        data_source: Multimodal dataset or directory
        n_epochs: Number of epochs
        n_prototypes: Number of ATL prototypes
        feature_dim: Feature dimension
        device: Device
        checkpoint_dir: Checkpoint directory
        a1_path: Optional path to pretrained A1 model
    
    Returns:
        Trained ATLSemanticHub
    """
    from modas.data.multimodal_dataset import (
        create_multimodal_dataloader,
        SyntheticMultimodalDataset
    )
    
    # Load pretrained V1
    v1 = V1SparseCoding(n_bases=feature_dim)
    v1_state = torch.load(v1_path, map_location=device)
    v1.load_state_dict(v1_state['model_state_dict'])
    
    # Load A1 if provided
    a1 = None
    if a1_path is not None:
        a1 = A1SparseCoding(n_bases=feature_dim)
        a1_state = torch.load(a1_path, map_location=device)
        a1.load_state_dict(a1_state['model_state_dict'])
    
    # Create language encoder
    language = LanguageEncoder(output_dim=feature_dim, load_pretrained=False)
    
    # Create ATL
    atl = ATLSemanticHub(n_prototypes=n_prototypes, feature_dim=feature_dim, **kwargs)
    
    # Create data
    if isinstance(data_source, str):
        dataloader = create_multimodal_dataloader(data_source)
        dataset = dataloader.dataset
    else:
        dataset = data_source
        dataloader = create_multimodal_dataloader(dataset)
    
    # Create test data
    test_data = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        test_data.append((sample.image, sample.text))
    
    # Train
    trainer = ATLTrainer(atl, v1, language, a1, device, checkpoint_dir)
    trainer.train(dataloader, test_data, n_epochs)
    trainer.visualize()
    
    return atl
