#!/usr/bin/env python3
"""
Run ablation experiments for MMFuse SData model.
Trains and evaluates models with different modality/component configurations.
Does NOT modify the original training code - uses config-driven ablation.

Usage:
  # Train one ablation (use --dataset for correct split before augmentation)
  python scripts/run_ablation.py --ablation no_audio --embeddings-dir embeddings/sdata_viscop --dataset dataset/sdata --epochs 10

  # List available ablations
  python scripts/run_ablation.py --list

  # Evaluate only (using existing checkpoint)
  python scripts/run_ablation.py --ablation no_audio --embeddings-dir embeddings/sdata_viscop --dataset dataset/sdata --eval-only
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

# Support both mmfuse package and direct imports (project root)
try:
    from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
    from mmfuse.fusion.multimodal_fusion import MultimodalFusion, MultimodalFusionWithAttention
except ModuleNotFoundError:
    from encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
    from fusion.multimodal_fusion import MultimodalFusion, MultimodalFusionWithAttention

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)


def load_ablation_config(ablation_name: str) -> dict:
    """Load ablation config from YAML."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "ablation_experiments.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Ablation config not found: {config_path}")
    with open(config_path) as f:
        all_configs = yaml.safe_load(f)
    if ablation_name not in all_configs:
        raise ValueError(
            f"Unknown ablation '{ablation_name}'. Available: {list(all_configs.keys())}"
        )
    return all_configs[ablation_name]


def collate_fn(batch):
    return batch


class PrecomputedSDataDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_dir: Path, config: dict):
        self.embeddings_dir = Path(embeddings_dir)
        self.num_classes = config.get("num_classes", 8)
        self.samples = list(sorted(self.embeddings_dir.glob("*.pt")))
        if not self.samples:
            raise RuntimeError(f"No .pt files in {embeddings_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx], map_location="cpu", weights_only=True)
        return {
            "vision_camera1": data["vision_camera1"],
            "vision_camera2": data["vision_camera2"],
            "audio": data["audio"],
            "target": data["target"],
        }


class ActionClassifier(nn.Module):
    def __init__(self, embedding_dim=256, num_classes=8):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MovementHead(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 3)

    def forward(self, x):
        return self.fc(x)


def build_embedding_ablation(batch, device, encoders, ablation_cfg):
    """Build embeddings with modality ablation (zero out disabled modalities)."""
    v1 = torch.stack([s["vision_camera1"] for s in batch]).to(device).float()
    v2 = torch.stack([s["vision_camera2"] for s in batch]).to(device).float()
    a = torch.stack([s["audio"] for s in batch]).to(device).float()
    for t in (v1, v2, a):
        t[~torch.isfinite(t)] = 0.0

    if not ablation_cfg.get("use_vision_cam1", True):
        v1 = torch.zeros_like(v1)
    if not ablation_cfg.get("use_vision_cam2", True):
        v2 = torch.zeros_like(v2)
    if not ablation_cfg.get("use_audio", True):
        a = torch.zeros_like(a)

    pressures = torch.zeros(len(batch), 2, device=device)
    emgs = torch.zeros(len(batch), 4, device=device)
    if not ablation_cfg.get("use_pressure", True):
        pressures = torch.zeros_like(pressures)
    if not ablation_cfg.get("use_emg", True):
        emgs = torch.zeros_like(emgs)

    p_emb = encoders["pressure"](pressures)
    e_emb = encoders["emg"](emgs)

    embeddings = {
        "vision_camera1": v1,
        "vision_camera2": v2,
        "audio": a,
        "pressure": p_emb,
        "emg": e_emb,
    }
    fusion = encoders["fusion"]
    if hasattr(fusion, "attention"):  # MultimodalFusionWithAttention
        fused, kl_losses = fusion(embeddings, return_kl=True)
    else:
        fused = fusion(embeddings)
        kl_losses = {}
    return fused, kl_losses


def _get_train_test_indices_split_before_aug(
    dataset_path: Path, cross_pair: bool, augment_variations: int, num_embedding_files: int
) -> Tuple[List[int], List[int]]:
    """
    Compute train/test indices with split BEFORE augmentation (by video pairs).
    Uses same sample order as precompute_sdata_embeddings. Requires --dataset.
    """
    import importlib.util
    _precompute_path = Path(__file__).resolve().parent / "precompute_sdata_embeddings.py"
    spec = importlib.util.spec_from_file_location("precompute_sdata", _precompute_path)
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    build_sample_list = _mod.build_sample_list

    # Full list in same order as precompute (split_before_aug=False)
    samples = build_sample_list(dataset_path, cross_pair, augment_variations, split_before_aug=False)
    if len(samples) != num_embedding_files:
        log.warning(
            "Sample count mismatch: dataset yields %d, embeddings have %d. Using dataset order.",
            len(samples),
            num_embedding_files,
        )
    # Split by (cam1, cam2) pairs
    all_pairs_with_label = []
    seen = set()
    for audio_path, cam1, cam2, label, v in samples:
        key = (str(cam1), str(cam2), label)
        if key not in seen:
            seen.add(key)
            all_pairs_with_label.append((str(cam1), str(cam2), label))
    unique_pairs = list({(p[0], p[1], p[2]) for p in all_pairs_with_label})
    pair_labels = [p[2] for p in unique_pairs]
    train_pairs, test_pairs = train_test_split(
        unique_pairs, test_size=0.1, stratify=pair_labels, random_state=42
    )
    train_set = set((p[0], p[1]) for p in train_pairs)
    test_set = set((p[0], p[1]) for p in test_pairs)

    train_idx = []
    test_idx = []
    for i, (audio_path, cam1, cam2, label, v) in enumerate(samples):
        pair_key = (str(cam1), str(cam2))
        if pair_key in train_set:
            train_idx.append(i)
        elif pair_key in test_set and v == 0:
            test_idx.append(i)
    return train_idx, test_idx


def load_movement_targets(config_path: str, num_classes: int) -> torch.Tensor:
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.exists():
        return torch.zeros(num_classes, 3)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    movements = cfg.get("movements", [])
    targets = []
    for i in range(num_classes):
        if i < len(movements):
            m = movements[i]
            targets.append(
                [m.get("delta_along", 0), m.get("delta_lateral", 0), m.get("magnitude", 0)]
            )
        else:
            targets.append([0, 0, 0])
    return torch.tensor(targets, dtype=torch.float32)


def train_ablation(args, ablation_cfg: dict):
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_dir = Path(args.embeddings_dir)
    with open(emb_dir / "config.json") as f:
        emb_config = json.load(f)
    vision_dim = emb_config["vision_dim"]
    audio_dim = emb_config.get("audio_dim", 768)
    num_classes = emb_config.get("num_classes", 8)

    ds = PrecomputedSDataDataset(emb_dir, emb_config)
    cross_pair = emb_config.get("cross_pair", False)
    augment_variations = emb_config.get("augment_variations", 16)

    if args.dataset:
        # Split BEFORE augmentation (by video pairs) - correct for no leakage
        train_idx, test_idx = _get_train_test_indices_split_before_aug(
            Path(args.dataset),
            cross_pair=cross_pair,
            augment_variations=augment_variations,
            num_embedding_files=len(ds.samples),
        )
        log.info("Split before augmentation: %d train | %d test (by video pairs)", len(train_idx), len(test_idx))
    else:
        # Fallback: split on flattened list (may leak - same pair in train and test)
        log.warning(
            "No --dataset provided. Using 90/10 split on flattened embeddings (possible leakage). "
            "Use --dataset path/to/sdata for correct split-before-augmentation."
        )
        labels = [
            torch.load(p, map_location="cpu", weights_only=True)["target"]
            for p in ds.samples
        ]
        try:
            train_idx, test_idx = train_test_split(
                range(len(ds)), test_size=0.1, stratify=labels, random_state=42
            )
        except ValueError:
            train_idx, test_idx = train_test_split(
                range(len(ds)), test_size=0.1, random_state=42
            )
    train_ds = Subset(ds, train_idx)
    test_ds = Subset(ds, test_idx)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    modality_dims = {
        "vision_camera1": vision_dim,
        "vision_camera2": vision_dim,
        "audio": audio_dim,
        "pressure": 256,
        "emg": 256,
    }

    pressure = PressureSensorEncoder(output_dim=256, input_features=2).to(device)
    emg = EMGSensorEncoder(output_dim=256, num_channels=3, input_features=4).to(device)

    fusion_type = ablation_cfg.get("fusion_type", "attention")
    if fusion_type == "concat":
        fusion = MultimodalFusion(
            modality_dims=modality_dims,
            fusion_dim=args.fusion_dim,
            fusion_method="concat_project",
            dropout=args.dropout,
        ).to(device)
    else:
        fusion = MultimodalFusionWithAttention(
            modality_dims=modality_dims,
            fusion_dim=args.fusion_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
        ).to(device)

    encoders = {"pressure": pressure, "emg": emg, "fusion": fusion}
    model = ActionClassifier(embedding_dim=args.fusion_dim, num_classes=num_classes).to(device)
    movement_head = MovementHead(embedding_dim=args.fusion_dim).to(device)

    use_movement = ablation_cfg.get("use_movement_head", True)
    use_kl = ablation_cfg.get("use_kl_loss", True)

    movement_targets = load_movement_targets(args.movement_config, num_classes).to(device)

    trainable = list(model.parameters()) + list(fusion.parameters())
    if use_movement:
        trainable += list(movement_head.parameters())
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    def build_fn(batch):
        return build_embedding_ablation(batch, device, encoders, ablation_cfg)

    log.info("Ablation: %s | %s", args.ablation, ablation_cfg.get("description", ""))
    log.info("Train: %d | Test: %d", len(train_idx), len(test_idx))

    history = {"train_loss": [], "train_acc": [], "test_acc": [], "epoch_sec": []}

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        model.train()
        fusion.train()
        movement_head.train()
        epoch_loss = 0.0
        epoch_kl = 0.0
        correct, total = 0, 0

        for batch in train_dl:
            targets = torch.tensor([s["target"] for s in batch], dtype=torch.long).to(device)
            fused, kl_losses = build_fn(batch)
            logits = model(fused)
            movement_pred = movement_head(fused)

            loss_bc = criterion(logits, targets)
            loss_kl = sum(kl_losses.values()) if (use_kl and kl_losses) else torch.tensor(0.0, device=device)
            loss_mov = (
                nn.functional.mse_loss(movement_pred, movement_targets[targets])
                if use_movement
                else torch.tensor(0.0, device=device)
            )
            loss = loss_bc + args.kl_weight * loss_kl + args.movement_weight * loss_mov

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss_bc.item()
            if kl_losses:
                epoch_kl += sum(v.item() for v in kl_losses.values())
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += len(batch)

        train_acc = correct / total if total > 0 else 0
        n_batches = len(train_dl)

        model.eval()
        fusion.eval()
        movement_head.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for batch in test_dl:
                targets = torch.tensor([s["target"] for s in batch], dtype=torch.long).to(device)
                fused, _ = build_fn(batch)
                logits = model(fused)
                test_correct += (logits.argmax(dim=1) == targets).sum().item()
                test_total += len(batch)
        test_acc = test_correct / test_total if test_total > 0 else 0
        model.train()
        fusion.train()
        movement_head.train()

        epoch_sec = time.perf_counter() - t0
        history["train_loss"].append(epoch_loss / n_batches)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["epoch_sec"].append(epoch_sec)

        log.info(
            "Epoch %d/%d | %.1fs | train_loss=%.4f train_acc=%.4f test_acc=%.4f",
            epoch + 1,
            args.epochs,
            epoch_sec,
            epoch_loss / n_batches,
            train_acc,
            test_acc,
        )

        ckpt = {
            "model_state": model.state_dict(),
            "fusion_state": fusion.state_dict(),
            "movement_state": movement_head.state_dict(),
            "movement_targets": movement_targets.cpu(),
            "num_classes": num_classes,
            "epoch": epoch + 1,
            "vision_dim": vision_dim,
            "fusion_dim": args.fusion_dim,
            "audio_encoder": emb_config.get("audio_encoder", "wav2vec"),
            "vision_encoder": emb_config.get("vision_encoder", "viscop"),
            "ablation": args.ablation,
            "ablation_config": ablation_cfg,
        }
        torch.save(ckpt, out_dir / f"ckpt_sdata_epoch_{epoch + 1}.pt")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "ablation_config.json", "w") as f:
        json.dump(ablation_cfg, f, indent=2)

    final_metrics = {
        "ablation": args.ablation,
        "test_accuracy": history["test_acc"][-1] if history["test_acc"] else 0,
        "train_accuracy": history["train_acc"][-1] if history["train_acc"] else 0,
        "final_epoch": args.epochs,
    }
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    log.info("Training complete. Checkpoints saved to %s", out_dir)
    return history


def run_evaluation(args, ablation_cfg: dict) -> dict:
    """Run evaluation on existing checkpoint. Returns metrics dict."""
    # we'll run the evaluate script as a subprocess or inline the key logic.
    # Actually evaluate_sdata has its own build_embedding - it doesn't support ablation.
    # So we need to either: 1) add ablation support to evaluate_sdata, or 2) inline eval here.
    # Let me add a simple eval that uses our build_embedding_ablation and computes accuracy.
    device = torch.device(args.device)
    emb_dir = Path(args.embeddings_dir)
    ckpt_path = Path(args.out_dir) / "ckpt_sdata_epoch_1.pt"
    # Find latest checkpoint
    ckpts = sorted(Path(args.out_dir).glob("ckpt_sdata_epoch_*.pt"))
    if not ckpts:
        log.error("No checkpoint found in %s. Run training first.", args.out_dir)
        return {}
    ckpt_path = ckpts[-1]

    with open(emb_dir / "config.json") as f:
        emb_config = json.load(f)
    vision_dim = emb_config["vision_dim"]
    audio_dim = emb_config.get("audio_dim", 768)
    num_classes = emb_config.get("num_classes", 8)

    ds = PrecomputedSDataDataset(emb_dir, emb_config)
    cross_pair = emb_config.get("cross_pair", False)
    augment_variations = emb_config.get("augment_variations", 16)
    if args.dataset:
        _, test_idx = _get_train_test_indices_split_before_aug(
            Path(args.dataset),
            cross_pair=cross_pair,
            augment_variations=augment_variations,
            num_embedding_files=len(ds.samples),
        )
    else:
        labels = [torch.load(p, map_location="cpu", weights_only=True)["target"] for p in ds.samples]
        try:
            _, test_idx = train_test_split(range(len(ds)), test_size=0.1, stratify=labels, random_state=42)
        except ValueError:
            _, test_idx = train_test_split(range(len(ds)), test_size=0.1, random_state=42)
    test_ds = Subset(ds, test_idx)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    modality_dims = {
        "vision_camera1": vision_dim,
        "vision_camera2": vision_dim,
        "audio": audio_dim,
        "pressure": 256,
        "emg": 256,
    }
    pressure = PressureSensorEncoder(output_dim=256, input_features=2).to(device)
    emg = EMGSensorEncoder(output_dim=256, num_channels=3, input_features=4).to(device)
    fusion_type = ablation_cfg.get("fusion_type", "attention")
    if fusion_type == "concat":
        fusion = MultimodalFusion(
            modality_dims=modality_dims,
            fusion_dim=args.fusion_dim,
            fusion_method="concat_project",
            dropout=args.dropout,
        ).to(device)
    else:
        fusion = MultimodalFusionWithAttention(
            modality_dims=modality_dims,
            fusion_dim=args.fusion_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
        ).to(device)
    model = ActionClassifier(embedding_dim=args.fusion_dim, num_classes=num_classes).to(device)
    encoders = {"pressure": pressure, "emg": emg, "fusion": fusion}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    fusion.load_state_dict(ckpt["fusion_state"])
    model.load_state_dict(ckpt["model_state"])

    fusion.eval()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_dl:
            targets = torch.tensor([s["target"] for s in batch], dtype=torch.long).to(device)
            fused, _ = build_embedding_ablation(batch, device, encoders, ablation_cfg)
            logits = model(fused)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += len(batch)
    test_acc = correct / total if total > 0 else 0
    metrics = {
        "ablation": args.ablation,
        "test_accuracy": test_acc,
        "num_test_samples": total,
        "checkpoint": str(ckpt_path),
    }
    results_path = Path(args.out_dir) / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Ablation %s: test_acc=%.4f (%d/%d)", args.ablation, test_acc, correct, total)
    log.info("Results saved to %s", results_path)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", help="Ablation experiment name (or --list to show all)")
    p.add_argument("--embeddings-dir", help="Path to precomputed embeddings")
    p.add_argument("--dataset", help="Path to raw sdata folder (required for correct train/test split before augmentation)")
    p.add_argument("--out-dir", default=None, help="Output dir (default: ablation_runs/<ablation>)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--fusion-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--kl-weight", type=float, default=0.1)
    p.add_argument("--movement-weight", type=float, default=0.5)
    p.add_argument("--movement-config", default="config/sdata_movement_config.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--list", action="store_true", help="List all ablation experiments and exit")
    p.add_argument("--eval-only", action="store_true", help="Only evaluate (skip training)")
    args = p.parse_args()

    if args.list:
        config_path = Path(__file__).resolve().parent.parent / "config" / "ablation_experiments.yaml"
        with open(config_path) as f:
            configs = yaml.safe_load(f)
        print("Available ablation experiments:\n")
        for name, cfg in configs.items():
            print(f"  {name}: {cfg.get('description', '')}")
        return 0

    if not args.ablation or not args.embeddings_dir:
        p.error("--ablation and --embeddings-dir are required (unless --list)")

    ablation_cfg = load_ablation_config(args.ablation)
    if args.out_dir is None:
        args.out_dir = str(Path(__file__).resolve().parent.parent / "ablation_runs" / args.ablation)
    args.out_dir = os.path.abspath(args.out_dir)

    if args.eval_only:
        run_evaluation(args, ablation_cfg)
    else:
        train_ablation(args, ablation_cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
