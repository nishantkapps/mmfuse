#!/usr/bin/env python3
"""
Minimal finetune: load precomputed .pt files, load model, train. No dataset class, no extra checks.

Usage:
  python -m mmfuse.training.finetune_simple --embeddings-dir embeddings/vima_bench --model-file checkpoints/model.pt --epochs 10 --batch-size 8
"""
import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_proj = Path(__file__).resolve().parent.parent
if str(_proj) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_proj))

from config_modality import FUSION_DIM, get_modality_dims, PRESSURE_DIM, EMG_DIM, TEXT_DIM, AUDIO_DIM
from mmfuse.training.finetune_model import MMFuseFinetuneModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--model-file", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    emb_dir = Path(args.embeddings_dir)
    model_path = Path(args.model_file)

    with open(emb_dir / "config.json") as f:
        config = json.load(f)
    pt_files = sorted(emb_dir.glob("*.pt"))
    vision_encoder = config.get("vision_encoder", "clip")
    default_dims = get_modality_dims(vision_encoder)
    vision_dim = config.get("vision_dim", default_dims["vision_camera1"])
    n_class = config.get("num_classes")
    if n_class is None:
        one = torch.load(pt_files[0], map_location="cpu", weights_only=True)
        n_class = one["target"] + 1
    log.info("Samples=%d classes=%d vision_dim=%d", len(pt_files), n_class, vision_dim)

    fusion_dim = FUSION_DIM
    if model_path.exists():
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        fusion_dim = ckpt.get("fusion_dim", FUSION_DIM)

    modality_dims = get_modality_dims(vision_encoder)
    modality_dims = {**modality_dims, "vision_camera1": vision_dim, "vision_camera2": vision_dim}
    model = MMFuseFinetuneModel(
        modality_dims=modality_dims,
        fusion_dim=fusion_dim,
        num_classes=n_class,
        use_movement_head=False,
    ).to(device)

    if model_path.exists():
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        model.fusion.load_state_dict(ckpt["fusion_state"], strict=False)
        if ckpt.get("num_classes") == n_class and "model_state" in ckpt:
            model.action_head.load_state_dict(ckpt["model_state"], strict=False)
        log.info("Loaded %s", model_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    n = len(pt_files)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        correct = 0
        total_samples = 0
        n_batches = 0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            batch = [torch.load(pt_files[i], map_location="cpu", weights_only=True) for i in idx.tolist()]
            B = len(batch)
            targets = torch.tensor([b["target"] for b in batch], dtype=torch.long).to(device)
            keys = [k for k in batch[0] if k != "target"]
            emb = {k: torch.stack([b[k] for b in batch]).to(device).float() for k in keys}
            if "pressure" not in emb:
                emb["pressure"] = torch.zeros(B, PRESSURE_DIM, device=device, dtype=torch.float32)
            if "emg" not in emb:
                emb["emg"] = torch.zeros(B, EMG_DIM, device=device, dtype=torch.float32)

            logits, _ = model(emb, return_kl=False)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == targets).sum().item()
            total_samples += B
            n_batches += 1

        acc = correct / total_samples if total_samples else 0
        avg_loss = total_loss / n_batches if n_batches else 0
        log.info("Epoch %d/%d loss=%.4f acc=%.2f%%", epoch + 1, args.epochs, avg_loss, 100 * acc)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.action_head.state_dict(),
        "fusion_state": model.fusion.state_dict(),
        "num_classes": n_class,
        "fusion_dim": fusion_dim,
    }, model_path)
    log.info("Saved %s", model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
