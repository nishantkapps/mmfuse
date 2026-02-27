#!/usr/bin/env python3
"""
Standalone inference for MMFuse SData model using current CLIP+Wav2Vec checkpoints.

Requires the mmfuse codebase (to reuse the exact fusion + sensor encoders as training),
plus a SData-style checkpoint (e.g. ckpt_sdata_epoch_20.pt or model_*_answer.pt)
and precomputed embeddings (e.g. embeddings/sdata_clip or embeddings/nextqa_clip).

Usage:
  python scripts/standalone_sdata_inference.py \
    --checkpoint mmfuse/checkpoints_clip_wav2vec_v3/ckpt_sdata_epoch_20.pt \
    --embeddings-dir embeddings/sdata_clip \
    --num-samples 10

Outputs: action class (0-7) and movement (delta_along, delta_lateral, magnitude) per sample.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

try:
    from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
    from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention
    from config_modality import (
        get_modality_dims,
        FUSION_DIM,
        AUDIO_DIM,
        TEXT_DIM,
        PRESSURE_DIM,
        EMG_DIM,
    )
except ImportError:
    # Fallback for running as a plain script from the repo root
    import sys as _sys

    _proj = Path(__file__).resolve().parent.parent
    if str(_proj) not in _sys.path:
        _sys.path.insert(0, str(_proj))
    from encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder  # type: ignore
    from fusion.multimodal_fusion import MultimodalFusionWithAttention  # type: ignore
    from config_modality import (  # type: ignore
        get_modality_dims,
        FUSION_DIM,
        AUDIO_DIM,
        TEXT_DIM,
        PRESSURE_DIM,
        EMG_DIM,
    )


class ActionClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 256, num_classes: int = 8):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MovementHead(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


COMMAND_NAMES = [
    "Start the Massage",
    "Focus Here",
    "Move down a little bit",
    "Go Back Up",
    "Stop. Pause for a second",
    "Move to the Left",
    "Move to the Right",
    "Right there, perfect spot",
]


def main():
    p = argparse.ArgumentParser(description="Standalone SData inference (CLIP+Wav2Vec MMFuse checkpoint)")
    p.add_argument("--checkpoint", required=True, help="Path to model .pth or .pt file")
    p.add_argument("--embeddings-dir", required=True, help="Path to precomputed embeddings (.pt files)")
    p.add_argument("--num-samples", type=int, default=10, help="Number of samples to run")
    p.add_argument("--output-json", help="Optional: save results to JSON file")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    emb_dir = Path(args.embeddings_dir)
    if not emb_dir.exists():
        print(f"Embeddings dir not found: {emb_dir}")
        return 1

    samples = sorted(emb_dir.glob("*.pt"))
    if not samples:
        print(f"No .pt files in {emb_dir}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    print("\nCHECKPOINT KEYS:")
    for k in ckpt.keys():
        print(" -", k)

    num_classes = ckpt.get("num_classes", 8)
    fusion_dim = ckpt.get("fusion_dim", FUSION_DIM)

    # Use embedding config to infer vision/audio dims when possible (e.g. sdata_clip, nextqa_clip)
    emb_config_path = emb_dir / "config.json"
    if emb_config_path.exists():
        with open(emb_config_path) as f:
            emb_config = json.load(f)
        vision_dim = emb_config.get("vision_dim", get_modality_dims(emb_config.get("vision_encoder", "clip"))["vision_camera1"])
        audio_dim = emb_config.get("audio_dim", AUDIO_DIM)
        num_classes = emb_config.get("num_classes", num_classes)
    else:
        dims = get_modality_dims("clip")
        vision_dim = dims["vision_camera1"]
        audio_dim = AUDIO_DIM

    modality_dims = {
        "vision_camera1": vision_dim,
        "vision_camera2": vision_dim,
        "audio": audio_dim,
        "text": TEXT_DIM,
        "pressure": PRESSURE_DIM,
        "emg": EMG_DIM,
    }

    pressure_enc = PressureSensorEncoder(output_dim=PRESSURE_DIM, input_features=2).to(device)
    emg_enc = EMGSensorEncoder(output_dim=EMG_DIM, num_channels=3, input_features=4).to(device)
    fusion = MultimodalFusionWithAttention(
        modality_dims=modality_dims,
        fusion_dim=fusion_dim,
        num_heads=8,
        dropout=0.2,
    ).to(device)
    classifier = ActionClassifier(embedding_dim=fusion_dim, num_classes=num_classes).to(device)
    movement_head = MovementHead(embedding_dim=fusion_dim).to(device)

    print("\nLoading flat checkpoint...")

    # Load fusion
    fusion_state = {k.replace("fusion.", ""): v 
                    for k, v in ckpt.items() 
                    if k.startswith("fusion.")}
    fusion.load_state_dict(fusion_state)

    # Load classifier (action head)
    classifier_state = {k.replace("action_head.", ""): v 
                        for k, v in ckpt.items() 
                        if k.startswith("action_head.")}
    classifier.load_state_dict(classifier_state)

    # Load movement head
    movement_state = {k.replace("movement_head.", ""): v 
                    for k, v in ckpt.items() 
                    if k.startswith("movement_head.")}

    has_movement = len(movement_state) > 0
    if has_movement:
        movement_head.load_state_dict(movement_state)

    print("✓ Fusion loaded")
    print("✓ Action head loaded")
    print("✓ Movement head loaded:", has_movement)

    fusion.eval()
    classifier.eval()
    movement_head.eval()

    n = min(args.num_samples, len(samples))
    results = []

    print("=" * 80)
    print("SDATA STANDALONE INFERENCE")
    print("=" * 80)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Movement head: {'Yes' if has_movement else 'No'}")
    print(f"Samples: {n}")
    print("=" * 80)

    for i, sample_path in enumerate(samples[:n]):
        data = torch.load(sample_path, map_location=device, weights_only=True)
        v1 = data["vision_camera1"].unsqueeze(0).float().to(device)
        v2 = data["vision_camera2"].unsqueeze(0).float().to(device)
        audio = data["audio"].unsqueeze(0).float().to(device)
        text_emb = data.get("text", torch.zeros(768)).unsqueeze(0).float().to(device)
        target = int(data["target"].item() if torch.is_tensor(data["target"]) else data["target"])

        v1[~torch.isfinite(v1)] = 0.0
        v2[~torch.isfinite(v2)] = 0.0
        audio[~torch.isfinite(audio)] = 0.0
        text_emb[~torch.isfinite(text_emb)] = 0.0

        p_emb = pressure_enc(torch.zeros(1, 2, device=device))
        e_emb = emg_enc(torch.zeros(1, 4, device=device))

        embeddings = {
            "vision_camera1": v1,
            "vision_camera2": v2,
            "audio": audio,
            "text": text_emb,
            "pressure": p_emb,
            "emg": e_emb,
        }

        with torch.no_grad():
            fused, _ = fusion(embeddings, return_kl=True)
            logits = classifier(fused)
            pred = logits.argmax(dim=1).item()
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            if has_movement:
                movement = movement_head(fused).cpu().numpy()[0]
                delta_along, delta_lateral, magnitude = float(movement[0]), float(movement[1]), float(movement[2])
            else:
                movement = None
                delta_along = delta_lateral = magnitude = None

        result = {
            "sample": sample_path.name,
            "ground_truth": int(target),
            "predicted_action": int(pred),
            "correct": pred == target,
        }
        if has_movement:
            result["location"] = {
                "delta_along": delta_along,
                "delta_lateral": delta_lateral,
                "magnitude": magnitude,
            }

        results.append(result)

        print(f"\n--- Sample {i + 1}: {sample_path.name} ---")
        print(f"  Ground truth:  action={target} ({COMMAND_NAMES[target] if target < len(COMMAND_NAMES) else '?'})")
        print(f"  Predicted:     action={pred} ({COMMAND_NAMES[pred] if pred < len(COMMAND_NAMES) else '?'})")
        print(f"  Correct:       {'Yes' if pred == target else 'No'}")
        print(f"  Action probs:  {[f'{p:.4f}' for p in probs]}")
        if has_movement:
            print(f"  Location:      delta_along={delta_along:.4f}, delta_lateral={delta_lateral:.4f}, magnitude={magnitude:.4f}")

    correct = sum(r["correct"] for r in results)
    print(f"\n{'=' * 80}")
    print(f"Accuracy: {correct}/{n} = {100 * correct / n:.1f}%")
    print("=" * 80)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"samples": results, "accuracy": correct / n}, f, indent=2)
        print(f"Results saved to {args.output_json}")

    return 0


if __name__ == "__main__":
    exit(main())
