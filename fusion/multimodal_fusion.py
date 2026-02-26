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
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.GELU(),
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
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.GELU()
            )
        
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final fusion: gated fusion over three core modalities (vision1, vision2, audio).
        # Gate network takes concat([h1, h2, h3]) and outputs 3 scalars.
        self.gate_net = nn.Linear(fusion_dim * 3, 3)
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
    
    def forward(self, embeddings: Dict[str, torch.Tensor], return_kl: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        Fuse with cross-modal attention and compute alignment losses.
        Pipeline:
          encoders -> projections -> cross-modal attention -> KL divergence (post-attn) ->
          3-way contrastive alignment (vision1, vision2, audio) -> concat/MLP -> fused.
        """
        # Project embeddings from each encoder into common fusion_dim space
        projected = {}
        for modality, embedding in embeddings.items():
            projected[modality] = self.projections[modality](embedding)

        # --- Cross-modal attention over core modalities ---
        # Focus attention on the three primary modalities we care about: two vision streams + audio.
        core_modalities = [k for k in projected if ('camera' in k or 'vision' in k or 'audio' in k)]
        modality_names = sorted(core_modalities) if core_modalities else sorted(projected.keys())
        # (batch_size, num_modalities, fusion_dim)
        stacked = torch.stack([projected[name] for name in modality_names], dim=1)
        attended, _ = self.attention(stacked, stacked, stacked)

        # Prepare per-modality attended embeddings for alignment losses
        name_to_index = {name: i for i, name in enumerate(modality_names)}
        kl_losses: dict = {}
        kl_temp = 2.0

        def _kl(p_src: torch.Tensor, p_tgt: torch.Tensor) -> torch.Tensor:
            q1 = torch.log_softmax(p_src / kl_temp, dim=-1)
            q2 = torch.softmax(p_tgt / kl_temp, dim=-1).clamp(min=1e-8)
            return torch.nn.functional.kl_div(q1, q2, reduction="batchmean")

        # Identify main modalities: two cameras + optional audio
        v1_key = next((k for k in modality_names if "camera1" in k or "vision_camera1" in k), None)
        v2_key = next((k for k in modality_names if "camera2" in k or "vision_camera2" in k), None)
        a_key = next((k for k in modality_names if "audio" in k), None)

        if v1_key is not None and v2_key is not None:
            v1_att = attended[:, name_to_index[v1_key], :]
            v2_att = attended[:, name_to_index[v2_key], :]
            # KL divergence between the two vision streams (post-attention)
            kl_cam = _kl(v1_att, v2_att)
            kl_losses["kl_camera"] = kl_cam

        # Optional text/audio KL if both present after attention
        t_key = next((k for k in modality_names if "text" in k), None)
        if t_key is not None and a_key is not None:
            t_att = attended[:, name_to_index[t_key], :]
            a_att = attended[:, name_to_index[a_key], :]
            kl_text_audio = _kl(t_att, a_att)
            kl_losses["kl_text_audio"] = kl_text_audio

        # --- 3-way contrastive alignment (vision1, vision2, audio) ---
        # Use attended embeddings as modality-specific representations for contrastive learning.
        if v1_key is not None and v2_key is not None and a_key is not None:
            v1 = attended[:, name_to_index[v1_key], :]  # (B, D)
            v2 = attended[:, name_to_index[v2_key], :]
            a = attended[:, name_to_index[a_key], :]

            def _contrastive(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
                """Symmetric InfoNCE between two modality views."""
                z1n = torch.nn.functional.normalize(z1, dim=-1)
                z2n = torch.nn.functional.normalize(z2, dim=-1)
                logits = (z1n @ z2n.t()) / temp  # (B, B)
                labels = torch.arange(z1.size(0), device=z1.device)
                loss_i = torch.nn.functional.cross_entropy(logits, labels)
                loss_j = torch.nn.functional.cross_entropy(logits.t(), labels)
                return 0.5 * (loss_i + loss_j)

            c_v1_v2 = _contrastive(v1, v2)
            c_v1_a = _contrastive(v1, a)
            c_v2_a = _contrastive(v2, a)
            kl_losses["contrastive_3way"] = (c_v1_v2 + c_v1_a + c_v2_a) / 3.0

        # --- Final fusion: gated fusion over (h1, h2, h3) = (vision1, vision2, audio) ---
        # Default fallback: average over attended modalities (if some core modality is missing).
        fused_vec = attended.mean(dim=1)
        if v1_key is not None and v2_key is not None and a_key is not None:
            h1 = attended[:, name_to_index[v1_key], :]  # (B, D)
            h2 = attended[:, name_to_index[v2_key], :]  # (B, D)
            h3 = attended[:, name_to_index[a_key], :]   # (B, D)
            concat_h = torch.cat([h1, h2, h3], dim=-1)  # (B, 3D)
            gates = torch.sigmoid(self.gate_net(concat_h))  # (B, 3)
            fused_vec = (
                gates[:, 0:1] * h1 +
                gates[:, 1:2] * h2 +
                gates[:, 2:3] * h3
            )
        fused = self.fusion_mlp(fused_vec)
        if return_kl:
            return fused, kl_losses
        return fused, {}
