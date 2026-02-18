"""
Multimodal Fusion Module
Combines encodings from all modalities (vision, audio, sensors) into unified representation
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List


class MultimodalFusion(nn.Module):
    """
    Fuses multiple modalities into a unified embedding space
    
    Takes embeddings from different modalities and projects them to
    a common dimension for joint processing and analysis.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        fusion_dim: int = 512,
        fusion_method: str = "concat_project",
        dropout: float = 0.2
    ):
        """
        Initialize Multimodal Fusion
        
        Args:
            modality_dims: Dictionary mapping modality names to their embedding dims
                          e.g., {'vision': 512, 'audio': 768, 'pressure': 256, 'emg': 256}
            fusion_dim: Target dimension for fused embeddings
            fusion_method: How to fuse modalities - 'concat_project', 'weighted_sum', 'bilinear'
            dropout: Dropout rate
        """
        super().__init__()
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        self.fusion_method = fusion_method
        
        # Create projection layers for each modality
        self.projections = nn.ModuleDict()
        for modality, input_dim in modality_dims.items():
            self.projections[modality] = nn.Sequential(
                nn.Linear(input_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Fusion-specific layers
        if fusion_method == "concat_project":
            # Project concatenated PROJECTED embeddings (not original dims)
            # After individual projections: each modality is fusion_dim
            concat_dim = len(modality_dims) * fusion_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(concat_dim, fusion_dim * 2),
                nn.BatchNorm1d(fusion_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim * 2, fusion_dim)
            )
        
        elif fusion_method == "weighted_sum":
            # Learnable attention weights for each modality
            num_modalities = len(modality_dims)
            self.attention_weights = nn.Parameter(
                torch.ones(num_modalities) / num_modalities
            )
        
        elif fusion_method == "bilinear":
            # Bilinear pooling for pairs of modalities (more complex)
            # Input is concatenated projected embeddings
            concat_dim = len(modality_dims) * fusion_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(concat_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU()
            )
        
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(
        self,
        embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse multimodal embeddings
        
        Args:
            embeddings: Dictionary mapping modality names to tensors
                       Each tensor should be of shape (batch_size, modality_dim)
        
        Returns:
            Fused embedding of shape (batch_size, fusion_dim)
        """
        # Project each modality to common dimension
        projected = {}
        for modality, embedding in embeddings.items():
            if modality not in self.projections:
                raise ValueError(f"Unknown modality: {modality}")
            projected[modality] = self.projections[modality](embedding)
        
        # Fuse based on selected method
        if self.fusion_method == "concat_project":
            # Concatenate all projected embeddings
            modality_names = sorted(projected.keys())  # Ensure consistent order
            concat_embedding = torch.cat(
                [projected[name] for name in modality_names],
                dim=1
            )
            fused = self.fusion_layer(concat_embedding)
        
        elif self.fusion_method == "weighted_sum":
            # Weighted sum of projections
            modality_names = sorted(projected.keys())
            fused = None
            for i, name in enumerate(modality_names):
                weight = torch.softmax(self.attention_weights, dim=0)[i]
                contrib = projected[name] * weight
                fused = contrib if fused is None else fused + contrib
        
        elif self.fusion_method == "bilinear":
            modality_names = sorted(projected.keys())
            concat_embedding = torch.cat(
                [projected[name] for name in modality_names],
                dim=1
            )
            fused = self.fusion_layer(concat_embedding)
        
        return fused
    
    def get_modality_embeddings(
        self,
        embeddings: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get projected embeddings for each modality
        (before fusion)
        
        Args:
            embeddings: Dictionary of raw embeddings from each encoder
        
        Returns:
            Dictionary of projected embeddings in common space
        """
        projected = {}
        for modality, embedding in embeddings.items():
            if modality not in self.projections:
                raise ValueError(f"Unknown modality: {modality}")
            projected[modality] = self.projections[modality](embedding)
        
        return projected


class MultimodalFusionWithAttention(nn.Module):
    """
    Advanced fusion with cross-modal attention mechanism
    Allows each modality to attend to other modalities
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        """
        Initialize Fusion with Attention
        
        Args:
            modality_dims: Dictionary of modality names to dimensions
            fusion_dim: Target fusion dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        
        # Projection layers for each modality
        self.projections = nn.ModuleDict()
        for modality, input_dim in modality_dims.items():
            self.projections[modality] = nn.Sequential(
                nn.Linear(input_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU()
            )
        
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final fusion: concat(attended) -> MLP. Use smaller hidden to reduce params.
        # fusion_dim * num_modalities can be large; hidden = fusion_dim keeps params low
        concat_dim = fusion_dim * len(modality_dims)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, embeddings: Dict[str, torch.Tensor], return_kl: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        Fuse with cross-modal attention and compute KL divergence for knowledge distillation.
        
        Args:
            embeddings: Dictionary of embeddings from each modality
            return_kl: Whether to return KL divergence values (default True)
        
        Returns:
            Tuple of (fused embedding, dict of KL divergences)
        """
        # Project embeddings
        projected = {}
        for modality, embedding in embeddings.items():
            projected[modality] = self.projections[modality](embedding)
        
        # --- Knowledge Distillation (KL Divergence) ---
        # Use temperature to soften distributions and avoid NaN from extreme log_softmax
        kl_temp = 2.0
        kl_losses = {}
        # Camera modalities: look for keys containing 'camera' or 'vision'
        camera_keys = [k for k in projected if 'camera' in k or 'vision' in k]
        if len(camera_keys) == 2:
            cam1, cam2 = camera_keys
            p1 = torch.log_softmax(projected[cam1] / kl_temp, dim=-1)
            p2 = torch.softmax(projected[cam2] / kl_temp, dim=-1).clamp(min=1e-8)
            kl_cam = torch.nn.functional.kl_div(p1, p2, reduction='batchmean')
            kl_losses['kl_camera'] = kl_cam
        # Text/Audio modalities
        text_keys = [k for k in projected if 'text' in k]
        audio_keys = [k for k in projected if 'audio' in k]
        if text_keys and audio_keys:
            text_key = text_keys[0]
            audio_key = audio_keys[0]
            p1 = torch.log_softmax(projected[text_key] / kl_temp, dim=-1)
            p2 = torch.softmax(projected[audio_key] / kl_temp, dim=-1).clamp(min=1e-8)
            kl_text_audio = torch.nn.functional.kl_div(p1, p2, reduction='batchmean')
            kl_losses['kl_text_audio'] = kl_text_audio
        
        # --- N-way cross alignment (cross-modal attention) ---
        modality_names = sorted(projected.keys())
        stacked = torch.stack(
            [projected[name] for name in modality_names],
            dim=1
        )  # (batch_size, num_modalities, fusion_dim)
        attended, _ = self.attention(stacked, stacked, stacked)
        batch_size = attended.size(0)
        flattened = attended.reshape(batch_size, -1)
        fused = self.fusion_mlp(flattened)
        if return_kl:
            return fused, kl_losses
        else:
            return fused
