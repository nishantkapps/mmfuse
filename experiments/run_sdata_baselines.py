#!/usr/bin/env python3
"""
Run baselines on SData and compare to MMFuse. Produces experimental numbers for the paper.

Baselines (train linear classifier on embeddings):
- Vision-only: VisCoP vision (cam1 + cam2) -> 8 classes
- Audio-only: Wav2Vec audio -> 8 classes
- Vision+Audio: Concat vision + audio -> 8 classes (simple fusion, no learned fusion)

MMFuse: Trained fusion + classifier (from checkpoint)

Output: experiments/results/sdata_comparison.csv, .tex

Usage:
  python experiments/run_sdata_baselines.py --checkpoint checkpoints/ckpt_sdata_epoch_5.pt
  python experiments/run_sdata_baselines.py  # auto-finds checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

try:
    from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
    from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention
except ModuleNotFoundError:
    from encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
    from fusion.multimodal_fusion import MultimodalFusionWithAttention

import yaml


def load_embeddings(emb_dir: Path):
    """Load all embeddings and labels from SData."""
    samples = sorted(emb_dir.glob("*.pt"))
    if not samples:
        raise RuntimeError(f"No .pt files in {emb_dir}")
    v1_list, v2_list, a_list, targets = [], [], [], []
    def to_np(t):
        return t.detach().numpy() if hasattr(t, "detach") else (t if isinstance(t, np.ndarray) else np.array(t))

    for p in samples:
        d = torch.load(p, map_location="cpu", weights_only=True)
        v1_list.append(to_np(d["vision_camera1"]))
        v2_list.append(to_np(d["vision_camera2"]))
        a_list.append(to_np(d["audio"]))
        targets.append(int(d["target"].item() if torch.is_tensor(d["target"]) else d["target"]))
    return (
        np.stack(v1_list),
        np.stack(v2_list),
        np.stack(a_list),
        np.array(targets),
    )


def run_baseline(X_train, y_train, X_test, y_test, name: str):
    """Train logistic regression, return dict with accuracy, precision, recall (macro)."""
    X_train = np.nan_to_num(np.asarray(X_train, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(np.asarray(X_test, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    acc = float(accuracy_score(y_test, y_pred))
    labels = np.unique(y_test)
    prec, rec, f1 = None, None, None
    if len(labels) <= 20:
        report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
        macro = report.get("macro avg", {})
        prec, rec = macro.get("precision"), macro.get("recall")
        f1 = macro.get("f1-score")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def run_mmfuse_eval(emb_dir: Path, ckpt_path: Path, train_idx, test_idx, device):
    """Run MMFuse (trained model) on test set. Returns dict with accuracy, precision, recall, movement_mse."""
    from experiments.run_dataset import (
        PrecomputedDataset,
        build_embedding,
        collate_fn,
    )
    from torch.utils.data import DataLoader, Subset

    with open(emb_dir / "config.json") as f:
        emb_config = json.load(f)
    vision_dim = emb_config.get("vision_dim", 3584)
    audio_dim = emb_config.get("audio_dim", 768)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    fusion_dim = ckpt["fusion_dim"]
    ckpt_num_classes = ckpt.get("num_classes", 8)

    ds = PrecomputedDataset(emb_dir, emb_config)
    test_ds = Subset(ds, test_idx)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    pressure = PressureSensorEncoder(output_dim=256, input_features=2).to(device)
    emg = EMGSensorEncoder(output_dim=256, num_channels=3, input_features=4).to(device)
    modality_dims = {
        "vision_camera1": vision_dim,
        "vision_camera2": vision_dim,
        "audio": audio_dim,
        "pressure": 256,
        "emg": 256,
    }
    fusion = MultimodalFusionWithAttention(
        modality_dims=modality_dims,
        fusion_dim=fusion_dim,
        num_heads=8,
        dropout=0.0,
    ).to(device)
    from experiments.run_dataset import ActionClassifier

    model = ActionClassifier(embedding_dim=fusion_dim, num_classes=ckpt_num_classes).to(device)
    encoders = {"pressure": pressure, "emg": emg, "fusion": fusion}

    fusion.load_state_dict(ckpt["fusion_state"])
    model.load_state_dict(ckpt["model_state"])
    fusion.eval()
    model.eval()

    y_true, y_pred = [], []
    movement_out = []
    has_movement = "movement_state" in ckpt
    if has_movement:
        from experiments.run_dataset import MovementHead

        mh = MovementHead(embedding_dim=fusion_dim).to(device)
        mh.load_state_dict(ckpt["movement_state"])
        mh.eval()

    with torch.no_grad():
        for batch in test_dl:
            targets = torch.tensor(
                [int(s["target"].item() if torch.is_tensor(s["target"]) else s["target"]) for s in batch],
                dtype=torch.long,
                device=device,
            )
            fused = build_embedding(batch, device, encoders)
            logits = model(fused)
            preds = logits.argmax(dim=1)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            if has_movement:
                movement_out.extend(mh(fused).cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred))
    labels = np.unique(y_true)
    prec, rec, f1 = None, None, None
    if len(labels) <= 20:
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        macro = report.get("macro avg", {})
        prec, rec = macro.get("precision"), macro.get("recall")
        f1 = macro.get("f1-score")

    movement_mse = None
    if has_movement and "movement_targets" in ckpt and movement_out:
        mt = ckpt["movement_targets"].cpu().numpy()
        if len(mt) >= 8 and all(t < len(mt) for t in y_true):
            movement_gt = np.array([mt[t] for t in y_true])
            movement_mse = float(np.mean((np.array(movement_out) - movement_gt) ** 2))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "movement_mse": movement_mse}


def find_checkpoint():
    proj = Path(__file__).resolve().parent.parent
    for pattern in ["checkpoints/ckpt_sdata_epoch_*.pt", "runs/*/ckpt_sdata_epoch_*.pt"]:
        c = sorted(proj.glob(pattern))
        if c:
            return str(c[-1])
    if (proj / "models/sdata_viscop/pytorch_model.bin").exists():
        return str(proj / "models/sdata_viscop/pytorch_model.bin")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-dir", default=None, help="embeddings/sdata_viscop")
    p.add_argument("--checkpoint", default=None, help="MMFuse checkpoint")
    p.add_argument("--out-dir", default=None, help="Output dir")
    args = p.parse_args()

    proj = Path(__file__).resolve().parent.parent
    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else proj / "embeddings" / "sdata_viscop"
    ckpt_path = args.checkpoint or find_checkpoint()
    out_dir = Path(args.out_dir) if args.out_dir else proj / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not emb_dir.exists():
        print(f"Embeddings not found: {emb_dir}")
        print("Run: python scripts/precompute_sdata_embeddings.py --dataset dataset/sdata --out-dir embeddings/sdata_viscop")
        return 1
    if not ckpt_path or not Path(ckpt_path).exists():
        print("No checkpoint found. Use --checkpoint path/to/ckpt_sdata_epoch_N.pt")
        return 1

    print("Loading SData embeddings...")
    v1, v2, a, y = load_embeddings(emb_dir)
    train_idx, test_idx = train_test_split(
        range(len(y)), test_size=0.2, stratify=y, random_state=42
    )

    # Feature matrices
    vision = np.concatenate([v1, v2], axis=1)  # (N, 7168)
    audio = a  # (N, 768)
    vision_audio = np.concatenate([vision, audio], axis=1)  # (N, 7936)

    X_tr_v, X_te_v = vision[train_idx], vision[test_idx]
    X_tr_a, X_te_a = audio[train_idx], audio[test_idx]
    X_tr_va, X_te_va = vision_audio[train_idx], vision_audio[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    print("Running baselines...")
    res_vision = run_baseline(X_tr_v, y_tr, X_te_v, y_te, "Vision-only")
    res_audio = run_baseline(X_tr_a, y_tr, X_te_a, y_te, "Audio-only")
    res_va = run_baseline(X_tr_va, y_tr, X_te_va, y_te, "Vision+Audio (concat)")

    print("Running MMFuse...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_mmfuse = run_mmfuse_eval(emb_dir, Path(ckpt_path), train_idx, test_idx, device)

    rows = [
        ("Vision-only (VisCoP)", res_vision["accuracy"], None),
        ("Audio-only (Wav2Vec)", res_audio["accuracy"], None),
        ("Vision+Audio (concat)", res_va["accuracy"], None),
        ("MMFuse (full)", res_mmfuse["accuracy"], res_mmfuse.get("movement_mse")),
    ]

    # CSV
    csv_path = out_dir / "sdata_comparison.csv"
    with open(csv_path, "w") as f:
        f.write("model,sdata_acc_pct,movement_mse\n")
        for name, acc, mse in rows:
            mse_str = f"{mse:.4f}" if mse is not None else ""
            f.write(f'"{name}",{100*acc:.2f},{mse_str}\n')
    print(f"Saved {csv_path}")

    # LaTeX
    tex_path = out_dir / "sdata_comparison.tex"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison on SData: MMFuse vs baselines. Vision-only and Audio-only use linear classifiers on VisCoP/Wav2Vec embeddings. Vision+Audio concatenates embeddings. MMFuse uses learned multimodal fusion (vision+audio+pressure+EMG) and movement head.}",
        r"\label{tab:sdata_comparison}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & SData Acc. (\%) & Movement MSE \\",
        r"\midrule",
    ]
    for name, acc, mse in rows:
        name_esc = name.replace("_", r"\_")
        mse_str = f"{mse:.4f}" if mse is not None else "---"
        if "MMFuse" in name:
            lines.append(rf"\textbf{{{name_esc}}} & {100*acc:.2f} & {mse_str} \\\\")
        else:
            lines.append(f"{name_esc} & {100*acc:.2f} & {mse_str} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {tex_path}")

    print("\n" + "=" * 60)
    print("SData Comparison (MMFuse vs Baselines)")
    print("=" * 60)
    print(f"{'Model':<30} {'SData Acc (%)':<14} {'Movement MSE'}")
    print("-" * 60)
    for name, acc, mse in rows:
        mse_str = f"{mse:.4f}" if mse is not None else "---"
        print(f"{name:<30} {100*acc:.2f}%{'':<8} {mse_str}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
