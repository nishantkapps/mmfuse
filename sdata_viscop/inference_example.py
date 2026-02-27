#!/usr/bin/env python3
"""Minimal inference example - run from this model directory."""
import argparse
import json
import sys
from pathlib import Path

# Add mmfuse project root (parent of models/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-dir", required=True, help="Path to precomputed embeddings")
    p.add_argument("--num-samples", type=int, default=5)
    args = p.parse_args()

    model_dir = Path(__file__).resolve().parent
    weights_path = model_dir / "pytorch_model.bin"
    config_path = model_dir / "config.json"

    if not weights_path.exists():
        print("pytorch_model.bin not found")
        sys.exit(1)

    config = json.load(open(config_path))
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Import after path setup
    from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
    from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention

    class ActionClassifier(torch.nn.Module):
        def __init__(self, embedding_dim, num_classes):
            super().__init__()
            self.fc = torch.nn.Linear(embedding_dim, num_classes)
        def forward(self, x):
            return self.fc(x)

    class MovementHead(torch.nn.Module):
        def __init__(self, embedding_dim):
            super().__init__()
            self.fc = torch.nn.Linear(embedding_dim, 3)
        def forward(self, x):
            return self.fc(x)

    fusion_dim = config["fusion_dim"]
    num_classes = config["num_classes"]
    vision_dim = config["vision_dim"]

    fusion = MultimodalFusionWithAttention(
        vision_dim=vision_dim, audio_dim=768, pressure_dim=256, emg_dim=256,
        output_dim=fusion_dim, num_heads=8, dropout=0.2
    )
    classifier = ActionClassifier(fusion_dim, num_classes)
    movement_head = MovementHead(fusion_dim) if config.get("has_movement_head") else None

    fusion.load_state_dict(ckpt["fusion_state"])
    classifier.load_state_dict(ckpt["model_state"])
    if movement_head and "movement_state" in ckpt:
        movement_head.load_state_dict(ckpt["movement_state"])

    fusion.eval()
    classifier.eval()
    if movement_head:
        movement_head.eval()

    # Load a few samples
    samples = sorted(Path(args.embeddings_dir).glob("*.pt"))[: args.num_samples]
    if not samples:
        print("No .pt files in", args.embeddings_dir)
        sys.exit(1)

    pressure_enc = PressureSensorEncoder(output_dim=256, input_features=2)
    emg_enc = EMGSensorEncoder(output_dim=256, num_channels=3, input_features=4)

    correct = 0
    for pth in samples:
        d = torch.load(pth, map_location="cpu", weights_only=True)
        v1 = d["vision_camera1"].unsqueeze(0).float()
        v2 = d["vision_camera2"].unsqueeze(0).float()
        a = d["audio"].unsqueeze(0).float()
        t = d["target"].item()
        v1[~torch.isfinite(v1)] = 0.0
        v2[~torch.isfinite(v2)] = 0.0
        a[~torch.isfinite(a)] = 0.0
        p_emb = pressure_enc(torch.zeros(1, 2))
        e_emb = emg_enc(torch.zeros(1, 4))
        emb = {"vision_camera1": v1, "vision_camera2": v2, "audio": a, "pressure": p_emb, "emg": e_emb}
        with torch.no_grad():
            fused = fusion(emb)
            logits = classifier(fused)
            pred = logits.argmax(dim=1).item()
            mov = movement_head(fused).squeeze().tolist() if movement_head else []
        correct += 1 if pred == t else 0
        mov_str = f" movement={mov}" if mov else ""
        print(f"{pth.name}: true={t} pred={pred}{mov_str}")

    print(f"\nAccuracy on {len(samples)} samples: {correct}/{len(samples)} = {100*correct/len(samples):.1f}%")

if __name__ == "__main__":
    main()
