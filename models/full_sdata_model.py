#!/usr/bin/env python3
"""
Full MMFuse model for SData-style training and export.

Goal
-----
- Build a **single nn.Module** that bundles:
  - Vision encoder (VisCoP or CLIP)
  - Audio encoder (Wav2Vec / Whisper / learnable)
  - Pressure + EMG encoders
  - Multimodal fusion (MultimodalFusionWithAttention)
  - Action head + optional movement head

- Provide a factory that can:
  1) Load your existing **SData checkpoint** (model.pt from train_sdata_attention.py),
  2) Instantiate encoders from their **pretrained sources** (VisCoP, Wav2Vec, etc.),
  3) Load fusion + heads from the checkpoint,
  4) Return a **single full model** whose `state_dict()` contains *all* weights.

Usage sketch
------------
    from mmfuse.models.full_sdata_model import FullSDataModel

    model = FullSDataModel.from_sdata_checkpoint(
        ckpt_path="checkpoints/model.pt",
        device="cuda",
        viscop_model_path="viscop_trained_models/viscop_qwen2.5_7b_viscop-lora_egocentric-expert",
    )

    # Now model.state_dict() contains encoders + fusion + heads
    torch.save(model.state_dict(), "full_sdata_model.pt")

Downstream (e.g. on another machine), you would:
    model = FullSDataModel(
        vision_encoder_kind="viscop",
        audio_encoder_kind="wav2vec",
        num_classes=8,
        fusion_dim=512,
    )
    state = torch.load("full_sdata_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)

This file **does not** depend on precomputed embeddings; it works at the
embedding level (expects tensors for modalities). How you read/resize
frames/audio is up to the caller (you can reuse VisionPreprocessor / AudioPreprocessor).
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from mmfuse.encoders.vision_encoder import VisionEncoder
    from mmfuse.encoders.vision_encoder_viscop import VisCoPVisionEncoder
    from mmfuse.encoders.audio_encoder import Wav2VecPooledEncoder
    from mmfuse.encoders.audio_encoder_whisper import WhisperAudioEncoder
    from mmfuse.encoders.audio_encoder_learnable import AudioEncoder as LearnableAudioEncoder
    from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
    from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention
    from config_modality import (
        FUSION_DIM,
        AUDIO_DIM,
        TEXT_DIM,
        PRESSURE_DIM,
        EMG_DIM,
        VISION_DIM_CLIP,
        VISION_DIM_VISCOP,
        get_modality_dims,
    )
except ImportError:  # pragma: no cover - direct script usage
    import sys
    _proj = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_proj))
    from encoders.vision_encoder import VisionEncoder
    from encoders.vision_encoder_viscop import VisCoPVisionEncoder
    from encoders.audio_encoder import Wav2VecPooledEncoder
    from encoders.audio_encoder_whisper import WhisperAudioEncoder
    from encoders.audio_encoder_learnable import AudioEncoder as LearnableAudioEncoder
    from encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
    from fusion.multimodal_fusion import MultimodalFusionWithAttention
    from config_modality import (
        FUSION_DIM,
        AUDIO_DIM,
        TEXT_DIM,
        PRESSURE_DIM,
        EMG_DIM,
        VISION_DIM_CLIP,
        VISION_DIM_VISCOP,
        get_modality_dims,
    )


class ActionClassifier(nn.Module):
    """Classification head: fused embedding -> num_classes."""

    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MovementHead(nn.Module):
    """Movement head: fused embedding -> (delta_along, delta_lateral, magnitude)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class FullSDataModel(nn.Module):
    """
    Full MMFuse model for SData-style multimodal inputs.

    Components:
      - vision_encoder: VisCoPVisionEncoder or VisionEncoder (CLIP)
      - audio_encoder: Wav2VecPooledEncoder / WhisperAudioEncoder / LearnableAudioEncoder
      - pressure_encoder: PressureSensorEncoder
      - emg_encoder: EMGSensorEncoder
      - fusion: MultimodalFusionWithAttention
      - action_head: ActionClassifier
      - movement_head: MovementHead (optional)

    Forward operates at the **embedding level**:
      - You call the encoders yourself (or reuse SData build_embedding logic)
        to obtain a dict:
            {
              "vision_camera1": (B, vision_dim),
              "vision_camera2": (B, vision_dim),
              "audio": (B, audio_dim),
              "text": (B, TEXT_DIM) or zeros,
              "pressure": (B, PRESSURE_DIM),
              "emg": (B, EMG_DIM),
            }
      - Then pass that dict to `forward_embeddings`.
    """

    def __init__(
        self,
        vision_encoder_kind: str = "viscop",   # "viscop" or "clip"
        audio_encoder_kind: str = "wav2vec",   # "wav2vec" | "whisper" | "learnable"
        fusion_dim: int = FUSION_DIM,
        num_classes: int = 8,
        use_movement_head: bool = True,
        viscop_model_path: str = "viscop_trained_models/viscop_qwen2.5_7b_viscop-lora_egocentric-expert",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.vision_encoder_kind = vision_encoder_kind
        self.audio_encoder_kind = audio_encoder_kind

        # 1) Encoders
        if vision_encoder_kind == "viscop":
            self.vision = VisCoPVisionEncoder(
                model_path=viscop_model_path,
                device=str(self.device),
                frozen=False,
            )
            vision_dim = VISION_DIM_VISCOP
        else:
            self.vision = VisionEncoder(device=str(self.device))
            vision_dim = VISION_DIM_CLIP

        if audio_encoder_kind == "wav2vec":
            self.audio = Wav2VecPooledEncoder(frozen=False, device=str(self.device))
        elif audio_encoder_kind == "whisper":
            self.audio = WhisperAudioEncoder(frozen=False, device=str(self.device))
        else:
            self.audio = LearnableAudioEncoder(device=str(self.device))

        self.pressure = PressureSensorEncoder(output_dim=PRESSURE_DIM, input_features=2)
        self.emg = EMGSensorEncoder(output_dim=EMG_DIM, num_channels=3, input_features=4)

        # 2) Fusion
        modality_dims = get_modality_dims("viscop" if vision_encoder_kind == "viscop" else "clip")
        # Override vision/audio dims with actual encoder dims
        modality_dims["vision_camera1"] = vision_dim
        modality_dims["vision_camera2"] = vision_dim
        modality_dims["audio"] = AUDIO_DIM  # Wav2Vec/Whisper/learnable project to AUDIO_DIM
        self.fusion = MultimodalFusionWithAttention(
            modality_dims=modality_dims,
            fusion_dim=fusion_dim,
            num_heads=8,
            dropout=0.2,
        )

        # 3) Heads
        self.action_head = ActionClassifier(embedding_dim=fusion_dim, num_classes=num_classes)
        self.use_movement_head = use_movement_head
        self.movement_head: Optional[MovementHead]
        if use_movement_head:
            self.movement_head = MovementHead(embedding_dim=fusion_dim)
        else:
            self.movement_head = None

        # Move modules to device
        self.to(self.device)

    @classmethod
    def from_sdata_checkpoint(
        cls,
        ckpt_path: str | Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        viscop_model_path: str = "viscop_trained_models/viscop_qwen2.5_7b_viscop-lora_egocentric-expert",
    ) -> "FullSDataModel":
        """
        Build a full model from an SData checkpoint produced by train_sdata_attention.py.

        - Instantiates encoders from their pretrained sources (VisCoP, Wav2Vec, etc.).
        - Loads fusion_state, model_state (action head), movement_state.
        - Returns a single nn.Module whose state_dict() contains **all** weights.
        """
        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        vision_enc = ckpt.get("vision_encoder", "viscop")
        audio_enc = ckpt.get("audio_encoder", "wav2vec")
        fusion_dim = ckpt.get("fusion_dim", FUSION_DIM)
        num_classes = ckpt.get("num_classes", 8)
        use_movement = ckpt.get("movement_state") is not None

        model = cls(
            vision_encoder_kind=vision_enc,
            audio_encoder_kind=audio_enc,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            use_movement_head=use_movement,
            viscop_model_path=viscop_model_path,
            device=device,
        )

        # Load fusion
        if "fusion_state" in ckpt:
            fusion_state = ckpt["fusion_state"]
            fusion_model_state = model.fusion.state_dict()
            filtered = {k: v for k, v in fusion_state.items() if k in fusion_model_state and fusion_model_state[k].shape == v.shape}
            model.fusion.load_state_dict(filtered, strict=False)

        # Load action head
        if "model_state" in ckpt and ckpt["model_state"] is not None:
            model.action_head.load_state_dict(ckpt["model_state"], strict=False)

        # Load movement head
        if use_movement and "movement_state" in ckpt and ckpt["movement_state"] is not None and model.movement_head is not None:
            model.movement_head.load_state_dict(ckpt["movement_state"], strict=False)

        return model

    def forward_embeddings(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward given **modality embeddings** through fusion + heads.

        Expected keys in `embeddings` (all tensors on same device/type):
          - vision_camera1: (B, vision_dim)
          - vision_camera2: (B, vision_dim)
          - audio: (B, AUDIO_DIM)
          - text: (B, TEXT_DIM)    [optional, zeros if missing]
          - pressure: (B, PRESSURE_DIM)
          - emg: (B, EMG_DIM)
        """
        device = self.device
        B = next(iter(embeddings.values())).shape[0]

        vision1 = embeddings.get("vision_camera1")
        vision2 = embeddings.get("vision_camera2")
        audio = embeddings.get("audio")
        text = embeddings.get("text", torch.zeros(B, TEXT_DIM, device=device, dtype=torch.float32))
        pressure = embeddings.get("pressure", torch.zeros(B, PRESSURE_DIM, device=device, dtype=torch.float32))
        emg = embeddings.get("emg", torch.zeros(B, EMG_DIM, device=device, dtype=torch.float32))

        emb = {
            "vision_camera1": vision1.to(device).float(),
            "vision_camera2": vision2.to(device).float(),
            "audio": audio.to(device).float(),
            "text": text.to(device).float(),
            "pressure": pressure.to(device).float(),
            "emg": emg.to(device).float(),
        }
        for k in emb:
            emb[k] = torch.nan_to_num(emb[k], nan=0.0, posinf=0.0, neginf=0.0)

        fused, _ = self.fusion(emb, return_kl=True)
        logits = self.action_head(fused)
        movement = self.movement_head(fused) if self.movement_head is not None else None
        return {"logits": logits, "movement": movement}


if __name__ == "__main__":  # pragma: no cover
    # Quick smoke test (build model from scratch, no checkpoint)
    model = FullSDataModel(
        vision_encoder_kind="viscop",
        audio_encoder_kind="wav2vec",
        fusion_dim=512,
        num_classes=8,
        use_movement_head=True,
    )
    B = 2
    emb = {
        "vision_camera1": torch.randn(B, VISION_DIM_VISCOP),
        "vision_camera2": torch.randn(B, VISION_DIM_VISCOP),
        "audio": torch.randn(B, AUDIO_DIM),
        "text": torch.randn(B, TEXT_DIM),
        "pressure": torch.randn(B, PRESSURE_DIM),
        "emg": torch.randn(B, EMG_DIM),
    }
    out = model.forward_embeddings(emb)
    print("logits:", out["logits"].shape, "movement:", None if out["movement"] is None else out["movement"].shape)

