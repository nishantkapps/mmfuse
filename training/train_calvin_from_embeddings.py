#!/usr/bin/env python3
"""
Train MMFuse fusion + a Calvin action head from precomputed CALVIN ABC embeddings.

Embeddings are produced by experiments/precompute_calvin_abc.py and live in
  embeddings/calvin_abc_clip/ (or a custom --embeddings-dir).

Each .pt file is expected to contain:
  - vision_camera1: CLIP embedding for base camera frame
  - vision_camera2: CLIP embedding for wrist camera frame
  - audio: zeros (no audio; kept for API consistency)
  - text: zeros
  - target: int task_index (not used as primary supervision here)
  - action: float tensor of shape (7,) :
        [delta_ee_pos(3), delta_ee_rot(3), gripper(1)]

This script:
  - Loads a SData-style checkpoint (fusion_state, num_classes, fusion_dim, etc.)
  - Rebuilds fusion with the same architecture and loads fusion_state
  - Adds a CalvinActionHead (fusion_dim -> 7) for continuous action prediction
  - Trains fusion + CalvinActionHead with MSE on the action vector

Classifier head from the SData checkpoint is not used; this is a pure action-regression
finetune of fusion on CALVIN ABC.

Usage example:

  python -m mmfuse.training.train_calvin_from_embeddings \\
    --checkpoint mmfuse/checkpoints_clip_wav2vec_v3/ckpt_sdata_epoch_20.pt \\
    --embeddings-dir embeddings/calvin_abc_clip \\
    --out mmfuse/checkpoints_clip_wav2vec_v3/model_calvin_actions.pt \\
    --epochs 20 --batch-size 64 --lr 1e-4 --lr-fusion 5e-5 --weight-decay 0.05
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

_proj = Path(__file__).resolve().parent.parent
import sys

if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

try:
    from mmfuse.training.finetune_dataset import PrecomputedFinetuneDataset, collate_precomputed
    from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention
    from config_modality import (
        get_modality_dims,
        FUSION_DIM,
        TEXT_DIM,
        PRESSURE_DIM,
        EMG_DIM,
        AUDIO_DIM,
    )
except ImportError:
    from finetune_dataset import PrecomputedFinetuneDataset, collate_precomputed  # type: ignore
    from fusion.multimodal_fusion import MultimodalFusionWithAttention  # type: ignore
    from config_modality import (  # type: ignore
        get_modality_dims,
        FUSION_DIM,
        TEXT_DIM,
        PRESSURE_DIM,
        EMG_DIM,
        AUDIO_DIM,
    )


class CalvinActionHead(nn.Module):
    """Regression head: fused embedding -> continuous action vector (7-dim by default)."""

    def __init__(self, embedding_dim: int, action_dim: int = 7):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CalvinTaskHead(nn.Module):
    """Classification head: fused embedding -> discrete task_index (Calvin tasks)."""

    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def main():
    p = argparse.ArgumentParser(description="Train Calvin action head from precomputed embeddings")
    p.add_argument("--checkpoint", required=True, help="Path to SData-style checkpoint (.pt)")
    p.add_argument(
        "--embeddings-dir",
        required=True,
        help="Path to precomputed CALVIN embeddings (from precompute_calvin_abc.py)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output path for fine-tuned checkpoint (default: calvin_<checkpoint_name>.pt)",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4, help="LR for action head")
    p.add_argument(
        "--lr-fusion",
        type=float,
        default=5e-5,
        help="LR for fusion (lower than head; default 5e-5)",
    )
    p.add_argument(
        "--weight-decay", type=float, default=0.05, help="Weight decay for AdamW optimizer"
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation fraction (0.1 -> 90/10 train/test split)",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples (for debugging)",
    )
    args = p.parse_args()

    cwd = Path.cwd()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = cwd / ckpt_path
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    emb_dir = Path(args.embeddings_dir)
    if not emb_dir.is_absolute():
        emb_dir = cwd / emb_dir
    if not emb_dir.exists():
        print(f"Embeddings dir not found: {emb_dir}")
        return 1

    config_path = emb_dir / "config.json"
    if not config_path.exists():
        print(f"No config.json in {emb_dir}")
        return 1
    with open(config_path) as f:
        emb_config = json.load(f)

    # Infer dims and num_classes from embeddings config
    vision_dim = emb_config.get(
        "vision_dim", get_modality_dims(emb_config.get("vision_encoder", "clip"))["vision_camera1"]
    )
    audio_dim = emb_config.get("audio_dim", AUDIO_DIM)
    num_classes = emb_config.get("num_classes", 8)
    modality_dims: Dict[str, int] = {
        "vision_camera1": vision_dim,
        "vision_camera2": vision_dim,
        "audio": audio_dim,
        "text": emb_config.get("text_dim", TEXT_DIM),
        "pressure": PRESSURE_DIM,
        "emg": EMG_DIM,
    }

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    fusion_dim = ckpt.get("fusion_dim", FUSION_DIM)

    ds = PrecomputedFinetuneDataset(emb_dir, max_samples=args.max_samples)
    if len(ds) == 0:
        print(f"No .pt files in {emb_dir}")
        return 1
    # Sanity: ensure 'action' key exists
    sample0 = ds[0]
    if "action" not in sample0:
        print("Embeddings do not contain 'action' key. Re-run precompute_calvin_abc.py first.")
        return 1

    # 90/10 train/test split by default (clamped to [0.05, 0.5])
    val_ratio = max(0.05, min(0.5, args.val_ratio))
    train_size = int((1.0 - val_ratio) * len(ds))
    val_size = len(ds) - train_size
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=generator)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_precomputed, num_workers=0
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_precomputed, num_workers=0
    )

    device = torch.device(args.device)
    fusion = MultimodalFusionWithAttention(
        modality_dims=modality_dims,
        fusion_dim=fusion_dim,
        num_heads=8,
        dropout=0.2,
    ).to(device)
    action_head = CalvinActionHead(embedding_dim=fusion_dim, action_dim=7).to(device)
    task_head = CalvinTaskHead(embedding_dim=fusion_dim, num_classes=num_classes).to(device)

    # Load fusion_state from SData checkpoint (partial load if shapes differ)
    if "fusion_state" in ckpt:
        state = ckpt["fusion_state"]
        model_state = fusion.state_dict()
        loaded = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
        if len(loaded) < len(state):
            print(f"Fusion: skipped {len(state) - len(loaded)} keys (shape mismatch)")
        fusion.load_state_dict(loaded, strict=False)
        print(f"Loaded fusion from {ckpt_path}")

    # Optimizer: separate LR for fusion and heads
    optimizer = torch.optim.AdamW(
        [
            {"params": fusion.parameters(), "lr": args.lr_fusion},
            {"params": action_head.parameters(), "lr": args.lr},
            {"params": task_head.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()

    # Training history for reporting
    history = {
        "train_loss": [],
        "train_task_acc": [],
        "val_loss": [],
        "val_task_acc": [],
    }

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        fusion.train()
        action_head.train()
        task_head.train()
        total_loss, n_batches = 0.0, 0
        correct_cls, total_cls = 0, 0

        for batch in train_dl:
            # Build embeddings dict, excluding 'target', 'action', 'reasoning_type'
            targets_action = batch["action"].to(device).float()
            targets_cls = batch["target"].to(device).long()
            B = targets_action.shape[0]
            emb = {
                k: batch[k].to(device).float()
                for k in batch
                if k not in ("target", "action", "reasoning_type") and isinstance(batch[k], torch.Tensor)
            }
            if "pressure" not in emb:
                emb["pressure"] = torch.zeros(B, PRESSURE_DIM, device=device, dtype=torch.float32)
            if "emg" not in emb:
                emb["emg"] = torch.zeros(B, EMG_DIM, device=device, dtype=torch.float32)
            if "text" not in emb:
                emb["text"] = torch.zeros(B, modality_dims["text"], device=device, dtype=torch.float32)
            for k in emb:
                emb[k] = torch.nan_to_num(emb[k], nan=0.0, posinf=0.0, neginf=0.0)

            optimizer.zero_grad()
            fused, _ = fusion(emb, return_kl=True)
            pred_action = action_head(fused)
            logits_task = task_head(fused)

            lambda_task = 0.01
            loss_reg = criterion(pred_action, targets_action)
            loss_cls = ce_criterion(logits_task, targets_cls)
            loss = loss_reg + lambda_task * loss_cls
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # classification accuracy
            preds = logits_task.argmax(dim=1)
            correct_cls += (preds == targets_cls).sum().item()
            total_cls += targets_cls.size(0)

        train_loss = total_loss / max(1, n_batches)
        train_acc = correct_cls / total_cls if total_cls else 0.0
        print(f"[Epoch {epoch+1}] train_loss={train_loss:.6f} train_task_acc={train_acc*100:.2f}%")
        history["train_loss"].append(train_loss)
        history["train_task_acc"].append(train_acc)

        # Validation
        fusion.eval()
        action_head.eval()
        task_head.eval()
        val_loss, n_val = 0.0, 0
        val_correct_cls, val_total_cls = 0, 0
        with torch.no_grad():
            for batch in val_dl:
                targets_action = batch["action"].to(device).float()
                targets_cls = batch["target"].to(device).long()
                B = targets_action.shape[0]
                emb = {
                    k: batch[k].to(device).float()
                    for k in batch
                    if k not in ("target", "action", "reasoning_type") and isinstance(batch[k], torch.Tensor)
                }
                if "pressure" not in emb:
                    emb["pressure"] = torch.zeros(B, PRESSURE_DIM, device=device, dtype=torch.float32)
                if "emg" not in emb:
                    emb["emg"] = torch.zeros(B, EMG_DIM, device=device, dtype=torch.float32)
                if "text" not in emb:
                    emb["text"] = torch.zeros(B, modality_dims["text"], device=device, dtype=torch.float32)
                for k in emb:
                    emb[k] = torch.nan_to_num(emb[k], nan=0.0, posinf=0.0, neginf=0.0)

                fused, _ = fusion(emb, return_kl=True)
                pred_action = action_head(fused)
                logits_task = task_head(fused)
                loss = criterion(pred_action, targets_action) + ce_criterion(logits_task, targets_cls)
                val_loss += loss.item()
                n_val += 1
                preds = logits_task.argmax(dim=1)
                val_correct_cls += (preds == targets_cls).sum().item()
                val_total_cls += targets_cls.size(0)

        val_loss = val_loss / max(1, n_val)
        val_acc = val_correct_cls / val_total_cls if val_total_cls else 0.0
        print(f"[Epoch {epoch+1}] val_loss={val_loss:.6f} val_task_acc={val_acc*100:.2f}%")
        history["val_loss"].append(val_loss)
        history["val_task_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "fusion_state": fusion.state_dict(),
                "calvin_action_head": action_head.state_dict(),
                "calvin_task_head": task_head.state_dict(),
                "fusion_dim": fusion_dim,
                "vision_dim": vision_dim,
                "audio_dim": audio_dim,
                "num_classes": num_classes,
            }

    # Save best model
    out_path = args.out
    if out_path is None:
        out_path = ckpt_path.parent / f"calvin_{ckpt_path.name}"
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = cwd / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if best_state is None:
        best_state = {
            "fusion_state": fusion.state_dict(),
            "calvin_action_head": action_head.state_dict(),
            "calvin_task_head": task_head.state_dict(),
            "fusion_dim": fusion_dim,
            "vision_dim": vision_dim,
            "audio_dim": audio_dim,
            "num_classes": num_classes,
        }

    torch.save(best_state, out_path)
    print(f"Saved Calvin action model to {out_path} (best val_loss={best_val_loss:.6f})")

    # Final detailed classification metrics on the held-out test set (val_ds)
    fusion.load_state_dict(best_state["fusion_state"])
    action_head.load_state_dict(best_state["calvin_action_head"])
    task_head.load_state_dict(best_state["calvin_task_head"])
    fusion.eval()
    task_head.eval()

    all_true, all_pred, all_proba = [], [], []
    with torch.no_grad():
        for batch in val_dl:
            targets_cls = batch["target"].to(device).long()
            B = targets_cls.shape[0]
            emb = {
                k: batch[k].to(device).float()
                for k in batch
                if k not in ("target", "action", "reasoning_type") and isinstance(batch[k], torch.Tensor)
            }
            if "pressure" not in emb:
                emb["pressure"] = torch.zeros(B, PRESSURE_DIM, device=device, dtype=torch.float32)
            if "emg" not in emb:
                emb["emg"] = torch.zeros(B, EMG_DIM, device=device, dtype=torch.float32)
            if "text" not in emb:
                emb["text"] = torch.zeros(B, modality_dims["text"], device=device, dtype=torch.float32)
            for k in emb:
                emb[k] = torch.nan_to_num(emb[k], nan=0.0, posinf=0.0, neginf=0.0)

            fused, _ = fusion(emb, return_kl=True)
            logits = task_head(fused)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            targets_np = targets_cls.cpu().numpy()

            all_true.extend(list(targets_np))
            all_pred.extend(list(preds))
            all_proba.extend(list(probs))

    all_true_tensor = torch.tensor(all_true)
    num_classes_eval = int(all_true_tensor.max().item()) + 1 if all_true else num_classes

    acc = accuracy_score(all_true, all_pred) if all_true else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="macro", zero_division=0
    ) if all_true else (0.0, 0.0, 0.0, None)
    try:
        auc = roc_auc_score(
            all_true,
            all_proba,
            multi_class="ovr",
            average="macro",
        )
    except Exception:
        auc = None

    print(
        f"Test (task_index) metrics: "
        f"acc={acc*100:.2f}% prec={prec*100:.2f}% rec={rec*100:.2f}% f1={f1*100:.2f}% "
        + (f"auc={auc:.4f}" if auc is not None else "auc=n/a")
    )

    # Save training history + final metrics alongside the model
    try:
        import json as _json

        hist_path = out_path.with_suffix(out_path.suffix + ".history.json")
        with open(hist_path, "w") as f:
            _json.dump(
                {
                    "train_loss": history.get("train_loss", []),
                    "train_task_acc": history.get("train_task_acc", []),
                    "val_loss": history.get("val_loss", []),
                    "val_task_acc": history.get("val_task_acc", []),
                    "best_val_loss": best_val_loss,
                    "test_accuracy": acc,
                    "test_precision_macro": prec,
                    "test_recall_macro": rec,
                    "test_f1_macro": f1,
                    "test_auc_macro_ovr": auc,
                    "epochs_run": len(history.get("train_loss", [])),
                    "num_classes": num_classes_eval,
                },
                f,
                indent=2,
            )
        print(f"Saved training history and test metrics to {hist_path}")
    except Exception as e:
        print(f"Could not save training history: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

