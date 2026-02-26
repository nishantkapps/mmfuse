#!/usr/bin/env python3
"""
Unified MMFuse model with integrated action + location (movement) heads.
Outputs: action logits, movement (delta_along, delta_lateral, magnitude).

Supports: vision, audio, optional text. Per-dataset fine-tuning.
Usage: source mmfuse-env/bin/activate && python -m mmfuse.training.finetune_model
"""
import torch
import torch.nn as nn
from pathlib import Path

try:
    from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention
except ImportError:
    import sys
    _proj = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_proj))
    from fusion.multimodal_fusion import MultimodalFusionWithAttention


class ActionClassifier(nn.Module):
    """Classification head: fused -> num_classes."""
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MovementHead(nn.Module):
    """Location head: fused -> (delta_along, delta_lateral, magnitude). Integrated in model."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MMFuseFinetuneModel(nn.Module):
    """
    Unified model: encoders -> fusion -> action_head + movement_head.
    Movement head is part of the model output (not a script on top).
    """

    def __init__(
        self,
        modality_dims: dict,
        fusion_dim: int = 512,
        num_classes: int = 8,
        num_heads: int = 8,
        dropout: float = 0.2,
        use_movement_head: bool = True,
    ):
        super().__init__()
        self.fusion = MultimodalFusionWithAttention(
            modality_dims=modality_dims,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.action_head = ActionClassifier(embedding_dim=fusion_dim, num_classes=num_classes)
        self.use_movement_head = use_movement_head
        if use_movement_head:
            self.movement_head = MovementHead(embedding_dim=fusion_dim)
        else:
            self.movement_head = None

    def forward(
        self,
        embeddings: dict,
        return_kl: bool = False,
        use_answer_head: bool = False,
    ):
        """Forward pass. use_answer_head: when True, same head is used as multiple-choice answer head (e.g. NextQA 5-way)."""
        fused, kl_losses = self.fusion(embeddings, return_kl=True)
        action_logits = self.action_head(fused)
        movement = self.movement_head(fused) if self.movement_head is not None else None
        if return_kl:
            return action_logits, movement, kl_losses
        return action_logits, movement


if __name__ == "__main__":
    # Quick test
    B = 4
    modality_dims = {"vision_camera1": 512, "vision_camera2": 512, "audio": 768}
    model = MMFuseFinetuneModel(modality_dims, fusion_dim=512, num_classes=8, use_movement_head=True)
    emb = {k: torch.randn(B, d) for k, d in modality_dims.items()}
    logits, mov, kl = model(emb, return_kl=True)
    print(f"action_logits: {logits.shape}, movement: {mov.shape if mov is not None else None}")
