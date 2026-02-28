#!/usr/bin/env python3
"""
Fine-tune using ONLY a saved model.pt (no VisCoP, CLIP, or any other encoder).
Loads: fusion + action head + optional movement head from model.pt.
Trains on: precomputed embeddings (same .pt format as SData / precompute scripts).
Saves: updated model.pt to --output (default: finetuned_<model_file>.pt).

Modes:
- Full finetune (default): train fusion + action head. Use this for high accuracy
  (paper-level ~80%+ NextQA, ~60%+ VideoMME/Charades). Fusion adapts to the QA task.
- Answer-head-only (--answer-head-only): freeze fusion; train only the answer head.
  Fast but limited accuracy (~40%) because fusion was trained for SData, not video QA.

Usage:
  # High accuracy: full finetune fusion + answer head (30–50 epochs recommended)
  python -m mmfuse.training.finetune_ckpt_only --model-file checkpoints/model.pt --embeddings-dir embeddings/nextqa_clip --output checkpoints/model_nextqa.pt --no-movement-head --epochs 40

  # Fast / low-resource: train only answer head (~40% NextQA)
  python -m mmfuse.training.finetune_ckpt_only ... --answer-head-only --no-movement-head --epochs 10
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

_proj = Path(__file__).resolve().parent.parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

try:
    from mmfuse.training.finetune_dataset import PrecomputedFinetuneDataset, collate_precomputed
    from mmfuse.training.finetune_model import MMFuseFinetuneModel
    from config_modality import FUSION_DIM, TEXT_DIM, PRESSURE_DIM, EMG_DIM
except ImportError:
    from finetune_dataset import PrecomputedFinetuneDataset, collate_precomputed
    from finetune_model import MMFuseFinetuneModel
    from config_modality import FUSION_DIM, TEXT_DIM, PRESSURE_DIM, EMG_DIM

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Fine-tune MMFuse from model.pt only (no external encoders)")
    p.add_argument("--model-file", required=True, help="Path to trained model.pt (SData or similar)")
    p.add_argument("--embeddings-dir", required=True, help="Path to precomputed embeddings (.pt + config.json)")
    p.add_argument("--output", default=None, help="Output path for fine-tuned model (default: finetuned_<model_file>.pt)")
    p.add_argument("--epochs", type=int, default=30, help="Epochs (use 30–50 for full finetune to reach 80%%+ on NextQA)")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate for head (and fusion if --lr-fusion not set)")
    p.add_argument("--lr-fusion", type=float, default=None, help="Learning rate for fusion (default: same as --lr). Lower e.g. 5e-5 can help stability.")
    p.add_argument("--scheduler", choices=["none", "cosine", "onecycle"], default="cosine", help="LR schedule (cosine recommended for full finetune)")
    p.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for optimizer")
    p.add_argument("--label-smoothing", type=float, default=0.15, help="Label smoothing for cross-entropy (set 0.0 to disable)")
    p.add_argument("--kl-weight", type=float, default=1e-4, help="Weight for KL regularization from fusion (0 to disable)")
    p.add_argument("--fusion-dropout", type=float, default=0.4, help="Dropout in fusion (higher reduces overfitting; try 0.4–0.5)")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of data for validation (0.2 = 80%% train / 20%% val)")
    p.add_argument("--early-stopping-patience", type=int, default=8, help="Stop if val acc does not improve for this many epochs (0 = disabled)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm for clipping (0 = no clip)")
    p.add_argument("--no-movement-head", action="store_true", help="Do not use movement head (e.g. NextQA/Charades/Video MME)")
    p.add_argument("--answer-head-only", action="store_true", help="Freeze fusion; train only answer head (fast but ~40%% accuracy). Omit for 80%%+.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-samples", type=int, default=None, help="Cap number of samples (for debugging)")
    args = p.parse_args()

    cwd = Path.cwd()
    model_file = Path(args.model_file)
    if not model_file.is_absolute():
        model_file = cwd / model_file
    if not model_file.exists():
        log.error("Model file not found: %s", model_file)
        return 1

    emb_dir = Path(args.embeddings_dir)
    if not emb_dir.is_absolute():
        emb_dir = cwd / emb_dir
    if not emb_dir.exists():
        log.error("Embeddings dir not found: %s", emb_dir)
        return 1

    config_path = emb_dir / "config.json"
    if not config_path.exists():
        log.error("No config.json in %s", emb_dir)
        return 1
    with open(config_path) as f:
        emb_config = json.load(f)

    # Modality dims from embeddings config (must match .pt keys)
    vision_dim = emb_config.get("vision_dim", 3584)
    audio_dim = emb_config.get("audio_dim", 768)
    num_classes = emb_config.get("num_classes", 8)
    modality_dims = {
        "vision_camera1": vision_dim,
        "vision_camera2": vision_dim,
        "audio": audio_dim,
        "text": emb_config.get("text_dim", TEXT_DIM),
        "pressure": PRESSURE_DIM,
        "emg": EMG_DIM,
    }

    # Load checkpoint to read fusion_dim and optional movement_targets
    ckpt = torch.load(model_file, map_location="cpu", weights_only=True)
    fusion_dim = ckpt.get("fusion_dim", FUSION_DIM)
    ckpt_num_classes = ckpt.get("num_classes", num_classes)
    use_movement = not args.no_movement_head and ckpt.get("movement_state") is not None
    answer_head_only = getattr(args, "answer_head_only", False)

    ds = PrecomputedFinetuneDataset(emb_dir, max_samples=args.max_samples)
    if len(ds) == 0:
        log.error("No .pt files in %s", emb_dir)
        return 1
    # num_classes: infer from dataset targets so answer head matches data (e.g. NextQA=5)
    inferred_classes = max(
        ds[i]["target"].item() if torch.is_tensor(ds[i]["target"]) else ds[i]["target"]
        for i in range(len(ds))
    ) + 1
    config_classes = emb_config.get("num_classes")
    num_classes = inferred_classes
    if config_classes is not None and config_classes >= inferred_classes:
        num_classes = config_classes
    elif config_classes is not None and config_classes != inferred_classes:
        log.warning("config.json num_classes=%s but data has %d classes; using %d", config_classes, inferred_classes, num_classes)
    log.info("Precomputed dataset: %s (%d samples, %d classes)%s", emb_dir, len(ds), num_classes, " [answer-head-only]" if answer_head_only else "")

    val_ratio = max(0.05, min(0.5, getattr(args, "val_ratio", 0.2)))
    train_size = int((1.0 - val_ratio) * len(ds))
    val_size = len(ds) - train_size
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=generator)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_precomputed, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_precomputed, num_workers=0)

    device = torch.device(args.device)
    fusion_dropout = getattr(args, "fusion_dropout", 0.4)
    model = MMFuseFinetuneModel(
        modality_dims=modality_dims,
        fusion_dim=fusion_dim,
        num_classes=num_classes,
        dropout=fusion_dropout,
        use_movement_head=use_movement,
    ).to(device)

    # Load only from model.pt (no other models)
    if "fusion_state" in ckpt:
        state = ckpt["fusion_state"]
        model_state = model.fusion.state_dict()
        loaded = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
        if len(loaded) < len(state):
            log.warning("Fusion: skipped %d keys (shape mismatch)", len(state) - len(loaded))
        model.fusion.load_state_dict(loaded, strict=False)
        log.info("Loaded fusion from %s", model_file)
    if answer_head_only:
        for p in model.fusion.parameters():
            p.requires_grad = False
        if model.movement_head is not None:
            for p in model.movement_head.parameters():
                p.requires_grad = False
        # Answer head: train from scratch for this dataset (num_classes may differ, e.g. NextQA=5)
        log.info("Answer-head-only: fusion and movement frozen; training new answer head (%d classes)", num_classes)
    else:
        if "model_state" in ckpt and ckpt["model_state"] is not None:
            action_state = ckpt["model_state"]
            head_state = model.action_head.state_dict()
            if list(action_state.keys()) == list(head_state.keys()) and all(action_state[k].shape == head_state[k].shape for k in head_state):
                model.action_head.load_state_dict(action_state, strict=True)
                log.info("Loaded action head from %s", model_file)
            else:
                log.warning("Action head shape/keys differ (checkpoint %d classes vs %d); head not loaded, training from scratch",
                            ckpt_num_classes, num_classes)
    if use_movement and "movement_state" in ckpt and ckpt["movement_state"] is not None:
        model.movement_head.load_state_dict(ckpt["movement_state"], strict=False)
        log.info("Loaded movement head from %s", model_file)

    # Optimizer: optional separate lr for fusion (full finetune only)
    if answer_head_only:
        optimizer = torch.optim.AdamW(model.action_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        lr_fusion = args.lr_fusion if args.lr_fusion is not None else args.lr
        fusion_params = list(model.fusion.parameters())
        head_params = list(model.action_head.parameters())
        if model.movement_head is not None:
            head_params += list(model.movement_head.parameters())
        optimizer = torch.optim.AdamW(
            [{"params": fusion_params, "lr": lr_fusion}, {"params": head_params, "lr": args.lr}],
            weight_decay=args.weight_decay,
        )
    n_steps = len(train_dl) * args.epochs
    if args.scheduler == "cosine" and not answer_head_only:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "onecycle" and not answer_head_only:
        max_lrs = [lr_fusion, args.lr] if args.lr_fusion is not None else args.lr
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lrs, total_steps=n_steps)
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    movement_targets = ckpt.get("movement_targets")
    if movement_targets is not None and use_movement:
        movement_targets = movement_targets.to(device)

    # Training history for analysis / plotting
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    patience = getattr(args, "early_stopping_patience", 8)

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_dl:
            targets = batch["target"].to(device)
            B = targets.shape[0]
            emb = {k: batch[k].to(device).float() for k in batch if k not in ("target", "reasoning_type") and isinstance(batch[k], torch.Tensor)}
            if "pressure" not in emb:
                emb["pressure"] = torch.zeros(B, PRESSURE_DIM, device=device, dtype=torch.float32)
            if "emg" not in emb:
                emb["emg"] = torch.zeros(B, EMG_DIM, device=device, dtype=torch.float32)
            if "text" not in emb:
                emb["text"] = torch.zeros(B, modality_dims["text"], device=device, dtype=torch.float32)
            for k in emb:
                emb[k] = torch.nan_to_num(emb[k], nan=0.0, posinf=0.0, neginf=0.0)

            optimizer.zero_grad()
            logits, movement, kl = model(emb, return_kl=True)
            loss = criterion(logits, targets)
            if not answer_head_only and kl is not None and args.kl_weight > 0.0:
                if isinstance(kl, dict):
                    kl_loss = sum(kl.values())
                elif isinstance(kl, (list, tuple)):
                    kl_loss = sum(kl)
                else:
                    kl_loss = kl
                loss = loss + args.kl_weight * kl_loss
            if not answer_head_only and movement is not None and movement_targets is not None and movement_targets.shape[0] >= num_classes:
                gt = movement_targets[targets]
                loss = loss + 0.5 * nn.functional.mse_loss(movement, gt)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            if scheduler is not None and args.scheduler == "onecycle":
                scheduler.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
        if scheduler is not None and args.scheduler == "cosine":
            scheduler.step()
        train_loss_epoch = total_loss / len(train_dl) if train_dl else 0.0
        train_acc = correct / total if total else 0.0
        history["train_loss"].append(train_loss_epoch)
        history["train_acc"].append(train_acc)
        log.info("Epoch %d train loss=%.4f acc=%.2f%%", epoch + 1, train_loss_epoch, 100 * train_acc)

        model.eval()
        correct_v, total_v = 0, 0
        with torch.no_grad():
            for batch in val_dl:
                targets = batch["target"].to(device)
                B = targets.shape[0]
                emb = {k: batch[k].to(device).float() for k in batch if k not in ("target", "reasoning_type") and isinstance(batch[k], torch.Tensor)}
                if "pressure" not in emb:
                    emb["pressure"] = torch.zeros(B, PRESSURE_DIM, device=device, dtype=torch.float32)
                if "emg" not in emb:
                    emb["emg"] = torch.zeros(B, EMG_DIM, device=device, dtype=torch.float32)
                if "text" not in emb:
                    emb["text"] = torch.zeros(B, modality_dims["text"], device=device, dtype=torch.float32)
                for k in emb:
                    emb[k] = torch.nan_to_num(emb[k], nan=0.0, posinf=0.0, neginf=0.0)
                logits, _, _ = model(emb, return_kl=True)
                correct_v += (logits.argmax(dim=1) == targets).sum().item()
                total_v += targets.size(0)
        val_acc = correct_v / total_v if total_v else 0.0
        history["val_acc"].append(val_acc)
        log.info("Epoch %d val acc=%.2f%%", epoch + 1, 100 * val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_state = {
                "fusion_state": model.fusion.state_dict(),
                "model_state": model.action_head.state_dict(),
                "movement_state": model.movement_head.state_dict() if model.movement_head is not None else None,
            }
        else:
            epochs_no_improve += 1
        if patience > 0 and epochs_no_improve >= patience:
            log.info("Early stopping: no val improvement for %d epochs (best val acc=%.2f%%)", patience, 100 * best_val_acc)
            break

    out_path = args.output
    if out_path is None:
        out_path = model_file.parent / f"finetuned_{model_file.name}"
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = cwd / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if best_state is not None:
        fusion_state = best_state["fusion_state"]
        model_state = best_state["model_state"]
        movement_state = best_state["movement_state"]
        log.info("Saving best model with val acc=%.2f%%", 100 * best_val_acc)
    else:
        fusion_state = model.fusion.state_dict()
        model_state = model.action_head.state_dict()
        movement_state = model.movement_head.state_dict() if model.movement_head is not None else None

    save_ckpt = {
        "fusion_state": fusion_state,
        "model_state": model_state,
        "movement_state": movement_state,
        "num_classes": num_classes,
        "fusion_dim": fusion_dim,
        "movement_targets": movement_targets.cpu() if movement_targets is not None else None,
    }
    torch.save(save_ckpt, out_path)
    log.info("Saved fine-tuned model to %s", out_path)
    # Save simple JSON history alongside the model (loss/acc per epoch)
    try:
        import json as _json
        hist_path = out_path.with_suffix(out_path.suffix + ".history.json")
        with open(hist_path, "w") as f:
            _json.dump(
                {
                    "train_loss": history.get("train_loss", []),
                    "train_acc": history.get("train_acc", []),
                    "val_acc": history.get("val_acc", []),
                    "best_val_acc": best_val_acc,
                    "epochs_run": len(history.get("train_loss", [])),
                },
                f,
                indent=2,
            )
        log.info("Saved training history to %s", hist_path)
    except Exception as e:
        log.warning("Could not save training history: %s", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
