#!/usr/bin/env python3
"""
Fine-tune MMFuse on real-world datasets (per-dataset, incremental).
Uses mmfuse-env. Location head trained jointly with action.

Usage:
  source /home/nishant/projects/mmfuse-env/bin/activate
  cd /home/nishant/projects/mmfuse
  python -m mmfuse.training.finetune --dataset nextqa --max-samples 500 --epochs 5
  python -m mmfuse.training.finetune --dataset video_mme --checkpoint checkpoints/ckpt.pt --epochs 3
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Ensure mmfuse package
_proj = Path(__file__).resolve().parent.parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

try:
    from mmfuse.training.finetune_dataset import FinetuneDataset, collate_finetune
    from mmfuse.training.finetune_model import MMFuseFinetuneModel
except ImportError:
    from finetune_dataset import FinetuneDataset, collate_finetune
    from finetune_model import MMFuseFinetuneModel

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def get_text_encoder_module(device, text_encoder="clip"):
    """Return (text encoder module, output_dim). Module is trainable."""
    if text_encoder == "clip":
        import open_clip
        full_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        # Use text tower only; wrap for batch encoding
        class CLIPTextEncoder(nn.Module):
            def __init__(self, clip_model, tokenizer, device):
                super().__init__()
                self.model = clip_model
                self.tokenizer = tokenizer
                self.device = device

            def forward(self, texts):
                t = self.tokenizer(texts).to(self.device)
                return self.model.encode_text(t).float()

        enc = CLIPTextEncoder(full_model, tokenizer, device).to(device)
        return enc, 512
    try:
        from mmfuse.encoders.text_encoder import TextEncoder
        enc = TextEncoder(output_dim=512, device=str(device))
        return enc, 512
    except Exception:
        return None, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nextqa", help="video_mme, nextqa, egoschema, charades, vima_bench")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vision-encoder", default="clip", choices=["clip", "viscop"])
    p.add_argument("--text-encoder", default="clip", help="clip or bert (mmfuse TextEncoder)")
    p.add_argument("--no-movement-head", action="store_true", help="Disable movement head (e.g. QA datasets)")
    p.add_argument("--freeze-vision", action="store_true", help="Keep vision encoder frozen (default: unfreeze for fine-tuning)")
    p.add_argument("--freeze-text", action="store_true", help="Keep text encoder frozen (default: unfreeze)")
    p.add_argument("--vision-lr", type=float, default=1e-5, help="LR for vision encoder (smaller than main)")
    p.add_argument("--text-lr", type=float, default=1e-5, help="LR for text encoder")
    args = p.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir) if args.data_dir else _proj / "extdataset" / args.dataset
    out_dir = Path(args.out_dir) if args.out_dir else _proj / "checkpoints" / f"finetune_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset %s from %s", args.dataset, data_dir)
    ds = FinetuneDataset(data_dir, max_samples=args.max_samples)
    if len(ds) == 0:
        log.error("No samples found")
        return 1

    n_class = max(s["target"] for s in ds.samples) + 1
    has_text = "text" in ds.samples[0]
    log.info("Samples: %d, classes: %d, has_text: %s", len(ds), n_class, has_text)

    train_size = int(0.9 * len(ds))
    train_ds, val_ds = random_split(ds, [train_size, len(ds) - train_size])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_finetune, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_finetune, num_workers=0)

    # Modality dims: vision (single or pair), optional text
    vision_dim = 3584 if args.vision_encoder == "viscop" else 512
    modality_dims = {"vision_camera1": vision_dim}
    if has_text:
        _, text_dim = get_text_encoder_module(device, args.text_encoder)
        modality_dims["text"] = text_dim

    model = MMFuseFinetuneModel(
        modality_dims=modality_dims,
        fusion_dim=512,
        num_classes=n_class,
        use_movement_head=not args.no_movement_head and args.dataset == "sdata",
    ).to(device)

    # Vision encoder (unfrozen by default for fine-tuning)
    if args.vision_encoder == "viscop":
        from mmfuse.encoders.vision_encoder_viscop import VisCoPVisionEncoder
        vision = VisCoPVisionEncoder(model_path="viscop_trained_models/viscop_qwen2.5_7b_viscop-lora_egocentric-expert", device=str(device))
    else:
        from mmfuse.encoders.vision_encoder import VisionEncoder
        vision = VisionEncoder(device=str(device), frozen=args.freeze_vision)
    vision = vision.to(device)
    if args.freeze_vision:
        for p in vision.parameters():
            p.requires_grad = False
        vision.eval()
        log.info("Vision encoder: frozen")
    else:
        vision.train()
        log.info("Vision encoder: trainable (lr=%s)", args.vision_lr)

    # Text encoder (unfrozen by default for QA datasets)
    text_encoder_module = None
    if has_text:
        text_encoder_module, _ = get_text_encoder_module(device, args.text_encoder)
        text_encoder_module = text_encoder_module.to(device)
        if args.freeze_text:
            for p in text_encoder_module.parameters():
                p.requires_grad = False
            text_encoder_module.eval()
            log.info("Text encoder: frozen")
        else:
            text_encoder_module.train()
            log.info("Text encoder: trainable (lr=%s)", args.text_lr)

    # Optimizer: param groups for vision, text, fusion+heads
    params = [{"params": model.parameters(), "lr": args.lr}]
    if not args.freeze_vision:
        params.append({"params": vision.parameters(), "lr": args.vision_lr})
    if not args.freeze_text and text_encoder_module is not None:
        params.append({"params": text_encoder_module.parameters(), "lr": args.text_lr})
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Load from checkpoint for incremental fine-tuning
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if "fusion_state" in ckpt:
            model.fusion.load_state_dict(ckpt["fusion_state"], strict=False)
            log.info("Loaded fusion from %s", args.checkpoint)
        if "model_state" in ckpt and ckpt.get("num_classes") == n_class:
            model.action_head.load_state_dict(ckpt["model_state"], strict=False)
            log.info("Loaded action head from %s", args.checkpoint)
        if "movement_state" in ckpt and model.movement_head is not None:
            model.movement_head.load_state_dict(ckpt["movement_state"], strict=False)
            log.info("Loaded movement head from %s", args.checkpoint)
        if "vision_state" in ckpt and not args.freeze_vision:
            vision.load_state_dict(ckpt["vision_state"], strict=False)
            log.info("Loaded vision encoder from %s", args.checkpoint)
        if "text_state" in ckpt and text_encoder_module is not None and not args.freeze_text:
            text_encoder_module.load_state_dict(ckpt["text_state"], strict=False)
            log.info("Loaded text encoder from %s", args.checkpoint)

    for epoch in range(args.epochs):
        model.train()
        if not args.freeze_vision:
            vision.train()
        if not args.freeze_text and text_encoder_module is not None:
            text_encoder_module.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_dl:
            frames = batch["frame"].to(device).float()
            targets = batch["target"].to(device)

            # CLIP expects ImageNet-normalized images (0-1 -> normalized)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=frames.device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=frames.device).view(1, 3, 1, 1)
            frames_norm = (frames - mean) / std
            v_emb = vision(frames_norm)
            if v_emb.dim() == 3:
                v_emb = v_emb.mean(dim=1)
            emb = {"vision_camera1": v_emb}
            if has_text and "text" in batch and text_encoder_module is not None:
                txt_emb = text_encoder_module(batch["text"])
                emb["text"] = txt_emb

            logits, mov, kl = model(emb, return_kl=True)
            loss = criterion(logits, targets)
            if kl:
                loss = loss + 0.1 * sum(kl.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == targets).sum().item()
            total += len(targets)

        acc = correct / total if total else 0
        log.info("Epoch %d/%d loss=%.4f acc=%.2f%%", epoch + 1, args.epochs, total_loss / len(train_dl), 100 * acc)

    ckpt = {
        "model_state": model.action_head.state_dict(),
        "fusion_state": model.fusion.state_dict(),
        "movement_state": model.movement_head.state_dict() if model.movement_head else None,
        "num_classes": n_class,
        "dataset": args.dataset,
    }
    if not args.freeze_vision:
        ckpt["vision_state"] = vision.state_dict()
    if not args.freeze_text and text_encoder_module is not None:
        ckpt["text_state"] = text_encoder_module.state_dict()
    torch.save(ckpt, out_dir / "checkpoint.pt")
    log.info("Saved to %s", out_dir / "checkpoint.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
