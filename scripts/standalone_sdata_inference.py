#!/usr/bin/env python3
"""
Standalone inference for MMFuse SData model.
No mmfuse codebase required - only PyTorch and NumPy.

Usage:
  python standalone_sdata_inference.py --checkpoint model.pth --embeddings-dir /path/to/embeddings [--num-samples 10]

Outputs: action class (0-7) and movement (delta_along, delta_lateral, magnitude) per sample.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

# --- Inlined model definitions (no mmfuse dependency) ---

class SensorEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class PressureSensorEncoder(SensorEncoder):
    def __init__(self, output_dim: int = 256, num_channels: int = 1, input_features: int = 100):
        super().__init__(input_dim=input_features, output_dim=output_dim, hidden_dim=128)
        self.num_channels = num_channels


class EMGSensorEncoder(SensorEncoder):
    def __init__(self, output_dim: int = 256, num_channels: int = 8, input_features: int = 100):
        super().__init__(input_dim=input_features, output_dim=output_dim, hidden_dim=128)
        self.num_channels = num_channels


class MultimodalFusionWithAttention(nn.Module):
    def __init__(self, modality_dims: Dict[str, int], fusion_dim: int = 512, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        self.projections = nn.ModuleDict()
        for modality, input_dim in modality_dims.items():
            self.projections[modality] = nn.Sequential(
                nn.Linear(input_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(),
            )
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        concat_dim = fusion_dim * len(modality_dims)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(self, embeddings: Dict[str, torch.Tensor], return_kl: bool = True) -> Tuple[torch.Tensor, dict]:
        projected = {mod: self.projections[mod](emb) for mod, emb in embeddings.items()}
        modality_names = sorted(projected.keys())
        stacked = torch.stack([projected[n] for n in modality_names], dim=1)
        attended, _ = self.attention(stacked, stacked, stacked)
        flattened = attended.reshape(attended.size(0), -1)
        fused = self.fusion_mlp(flattened)
        kl_losses = {}
        if return_kl:
            return fused, kl_losses
        return fused


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
    p = argparse.ArgumentParser(description="Standalone SData inference - no mmfuse required")
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

    num_classes = ckpt.get("num_classes", 8)
    fusion_dim = ckpt.get("fusion_dim", 256)
    vision_dim = ckpt.get("vision_dim", 3584)
    audio_dim = 768

    # Try to load embedding config for vision_dim
    emb_config_path = emb_dir / "config.json"
    if emb_config_path.exists():
        with open(emb_config_path) as f:
            emb_config = json.load(f)
        vision_dim = emb_config.get("vision_dim", vision_dim)
        num_classes = emb_config.get("num_classes", num_classes)

    modality_dims = {
        "vision_camera1": vision_dim,
        "vision_camera2": vision_dim,
        "audio": audio_dim,
        "pressure": 256,
        "emg": 256,
    }

    pressure_enc = PressureSensorEncoder(output_dim=256, input_features=2).to(device)
    emg_enc = EMGSensorEncoder(output_dim=256, num_channels=3, input_features=4).to(device)
    fusion = MultimodalFusionWithAttention(
        modality_dims=modality_dims,
        fusion_dim=fusion_dim,
        num_heads=8,
        dropout=0.2,
    ).to(device)
    classifier = ActionClassifier(embedding_dim=fusion_dim, num_classes=num_classes).to(device)
    movement_head = MovementHead(embedding_dim=fusion_dim).to(device)

    fusion.load_state_dict(ckpt["fusion_state"])
    classifier.load_state_dict(ckpt["model_state"])
    has_movement = "movement_state" in ckpt
    if has_movement:
        movement_head.load_state_dict(ckpt["movement_state"])

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
        target = int(data["target"].item() if torch.is_tensor(data["target"]) else data["target"])

        v1[~torch.isfinite(v1)] = 0.0
        v2[~torch.isfinite(v2)] = 0.0
        audio[~torch.isfinite(audio)] = 0.0

        p_emb = pressure_enc(torch.zeros(1, 2, device=device))
        e_emb = emg_enc(torch.zeros(1, 4, device=device))

        embeddings = {
            "vision_camera1": v1,
            "vision_camera2": v2,
            "audio": audio,
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
