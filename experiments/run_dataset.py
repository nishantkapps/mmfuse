#!/usr/bin/env python3
"""
Run MMFuse model evaluation on a single cross-dataset benchmark.
Expects precomputed embeddings in embeddings/<dataset>/ with config.json and *.pt files.
Format: each .pt has vision_camera1, vision_camera2, audio (or text as 768-dim), target.

Evaluation modes:
- zero-shot: Use SData-trained classifier (only works for SData; cross-dataset gets ~0-25%).
- linear-probe: Train a linear classifier on fused embeddings (measures representation transfer).
  Use for cross-dataset to get meaningful accuracy (can approach VisCoP-level transfer).

Usage:
  python experiments/run_dataset.py --dataset video_mme --checkpoint path/to/model.pt --linear-probe
  python experiments/run_dataset.py --dataset sdata --checkpoint path/to/model.pt  # zero-shot for SData
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

try:
    from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
    from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention
except ModuleNotFoundError:
    from encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
    from fusion.multimodal_fusion import MultimodalFusionWithAttention

import yaml


def collate_fn(batch):
    return batch


class PrecomputedDataset(torch.utils.data.Dataset):
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


def build_embedding(batch, device, encoders):
    v1 = torch.stack([s["vision_camera1"] for s in batch]).to(device).float()
    v2 = torch.stack([s["vision_camera2"] for s in batch]).to(device).float()
    a = torch.stack([s["audio"] for s in batch]).to(device).float()
    for t in (v1, v2, a):
        t[~torch.isfinite(t)] = 0.0
    pressures = torch.zeros(len(batch), 2, device=device)
    emgs = torch.zeros(len(batch), 4, device=device)
    p_emb = encoders["pressure"](pressures)
    e_emb = encoders["emg"](emgs)
    embeddings = {
        "vision_camera1": v1,
        "vision_camera2": v2,
        "audio": a,
        "pressure": p_emb,
        "emg": e_emb,
    }
    fused, _ = encoders["fusion"](embeddings, return_kl=True)
    return fused


class ActionClassifier(torch.nn.Module):
    def __init__(self, embedding_dim=256, num_classes=8):
        super().__init__()
        self.fc = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MovementHead(torch.nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.fc = torch.nn.Linear(embedding_dim, 3)

    def forward(self, x):
        return self.fc(x)


def run_evaluation(emb_dir: Path, ckpt_path: Path, out_dir: Path, num_classes: int, device):
    with open(emb_dir / "config.json") as f:
        emb_config = json.load(f)
    vision_dim = emb_config.get("vision_dim", 3584)
    audio_dim = emb_config.get("audio_dim", 768)
    dataset_num_classes = emb_config.get("num_classes", num_classes)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    ckpt_num_classes = ckpt.get("num_classes", 8)
    fusion_dim = ckpt["fusion_dim"]

    ds = PrecomputedDataset(emb_dir, emb_config)
    labels = [torch.load(p, map_location="cpu", weights_only=True)["target"] for p in ds.samples]
    try:
        train_idx, test_idx = train_test_split(
            range(len(ds)), test_size=0.1, stratify=labels, random_state=42
        )
    except ValueError:
        train_idx, test_idx = train_test_split(range(len(ds)), test_size=0.1, random_state=42)
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
        dropout=0.2,
    ).to(device)
    model = ActionClassifier(embedding_dim=fusion_dim, num_classes=ckpt_num_classes).to(device)
    encoders = {"pressure": pressure, "emg": emg, "fusion": fusion}

    fusion.load_state_dict(ckpt["fusion_state"])
    model.load_state_dict(ckpt["model_state"])

    fusion.eval()
    model.eval()
    y_true, y_pred = [], []
    movement_outputs = []
    has_movement = "movement_state" in ckpt
    if has_movement:
        movement_head = MovementHead(embedding_dim=fusion_dim).to(device)
        movement_head.load_state_dict(ckpt["movement_state"])
        movement_head.eval()

    with torch.no_grad():
        for batch in test_dl:
            targets = torch.tensor(
                [int(s["target"].item() if torch.is_tensor(s["target"]) else s["target"]) for s in batch],
                dtype=torch.long,
                device=device,
            )
            fused = build_embedding(batch, device, encoders)
            logits = model(fused)
            if dataset_num_classes < ckpt_num_classes:
                logits = logits[:, :dataset_num_classes]
            preds = logits.argmax(dim=1)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            if has_movement:
                mov = movement_head(fused).cpu().numpy()
                movement_outputs.extend(mov.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = float(np.mean(y_true == y_pred))

    # Per-class metrics (accuracy, precision, recall, f1) for VisCoP-style tables
    per_class = {}
    if dataset_num_classes is not None and dataset_num_classes <= 20:
        report = classification_report(
            y_true, y_pred, labels=range(dataset_num_classes),
            output_dict=True, zero_division=0
        )
        for i in range(dataset_num_classes):
            key = str(i)
            if key in report:
                per_class[key] = {
                    "precision": report[key]["precision"],
                    "recall": report[key]["recall"],
                    "f1-score": report[key]["f1-score"],
                    "support": int(report[key]["support"]),
                }
        # Per-class accuracy (correct / total for that class)
        per_class_acc = {}
        for c in range(dataset_num_classes):
            mask = y_true == c
            if mask.sum() > 0:
                per_class_acc[str(c)] = float(np.mean(y_pred[mask] == c))
        for k, v in per_class_acc.items():
            if k in per_class:
                per_class[k]["accuracy"] = v

    results = {
        "dataset": "",
        "accuracy": accuracy,
        "num_samples": len(y_true),
        "num_classes": dataset_num_classes,
        "has_movement_head": has_movement,
        "movement_mse": None,
        "per_class": per_class if per_class else None,
    }
    # Movement MSE only for SData (8 classes); movement_targets has 8 rows
    if has_movement and "movement_targets" in ckpt and movement_outputs:
        mt = ckpt["movement_targets"].cpu().numpy()
        mt_len = len(mt)
        valid = np.array([t < mt_len for t in y_true])
        if valid.all():
            movement_pred = np.array(movement_outputs)
            movement_gt = np.array([mt[t] for t in y_true])
            mse = float(np.mean((movement_pred - movement_gt) ** 2))
            results["movement_mse"] = mse

    return results


def run_evaluation_linear_probe(emb_dir: Path, ckpt_path: Path, out_dir: Path, num_classes: int, device):
    """Train a linear classifier on fused embeddings. Measures representation transfer (can approach VisCoP)."""
    with open(emb_dir / "config.json") as f:
        emb_config = json.load(f)
    vision_dim = emb_config.get("vision_dim", 3584)
    audio_dim = emb_config.get("audio_dim", 768)
    dataset_num_classes = emb_config.get("num_classes", num_classes)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    fusion_dim = ckpt["fusion_dim"]

    ds = PrecomputedDataset(emb_dir, emb_config)
    labels = [torch.load(p, map_location="cpu", weights_only=True)["target"] for p in ds.samples]
    try:
        train_idx, test_idx = train_test_split(
            range(len(ds)), test_size=0.2, stratify=labels, random_state=42
        )
    except ValueError:
        train_idx, test_idx = train_test_split(range(len(ds)), test_size=0.2, random_state=42)

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
    fusion.load_state_dict(ckpt["fusion_state"])
    fusion.eval()
    encoders = {"pressure": pressure, "emg": emg, "fusion": fusion}

    train_dl = DataLoader(Subset(ds, train_idx), batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(Subset(ds, test_idx), batch_size=32, shuffle=False, collate_fn=collate_fn)

    X_train, y_train = [], []
    with torch.no_grad():
        for batch in train_dl:
            fused = build_embedding(batch, device, encoders)
            X_train.append(fused.cpu().numpy())
            y_train.extend([int(s["target"].item() if torch.is_tensor(s["target"]) else s["target"]) for s in batch])
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    X_test, y_test = [], []
    with torch.no_grad():
        for batch in test_dl:
            fused = build_embedding(batch, device, encoders)
            X_test.append(fused.cpu().numpy())
            y_test.extend([int(s["target"].item() if torch.is_tensor(s["target"]) else s["target"]) for s in batch])
    X_test = np.vstack(X_test)
    y_test = np.array(y_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    accuracy = float(np.mean(y_test == y_pred))

    per_class = {}
    if dataset_num_classes is not None and dataset_num_classes <= 20:
        report = classification_report(
            y_test, y_pred, labels=range(dataset_num_classes),
            output_dict=True, zero_division=0
        )
        for i in range(dataset_num_classes):
            key = str(i)
            if key in report:
                per_class[key] = {
                    "precision": report[key]["precision"],
                    "recall": report[key]["recall"],
                    "f1-score": report[key]["f1-score"],
                    "support": int(report[key]["support"]),
                }
        per_class_acc = {}
        for c in range(dataset_num_classes):
            mask = y_test == c
            if mask.sum() > 0:
                per_class_acc[str(c)] = float(np.mean(y_pred[mask] == c))
        for k, v in per_class_acc.items():
            if k in per_class:
                per_class[k]["accuracy"] = v

    return {
        "dataset": "",
        "accuracy": accuracy,
        "num_samples": len(y_test),
        "num_classes": dataset_num_classes,
        "has_movement_head": False,
        "movement_mse": None,
        "per_class": per_class if per_class else None,
        "eval_mode": "linear_probe",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Dataset name: video_mme, nextqa, charades, egoschema, vima_bench, sdata")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--embeddings-dir", default=None, help="Override embeddings path")
    p.add_argument("--out-dir", default=None, help="Override output path")
    p.add_argument("--linear-probe", action="store_true",
                   help="Train linear classifier on fused embeddings (for cross-dataset transfer; default for video_mme/nextqa/charades/egoschema)")
    p.add_argument("--zero-shot", action="store_true",
                   help="Use SData-trained classifier (only meaningful for SData)")
    args = p.parse_args()

    config_path = Path(__file__).resolve().parent / "config" / "datasets.yaml"
    with open(config_path) as f:
        configs = yaml.safe_load(f)
    if args.dataset not in configs:
        print(f"Unknown dataset: {args.dataset}. Available: {list(configs.keys())}")
        return 1

    cfg = configs[args.dataset]
    proj = Path(__file__).resolve().parent.parent
    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else proj / cfg["embeddings_dir"]
    out_dir = Path(args.out_dir) if args.out_dir else proj / "experiments" / "results" / args.dataset

    if not emb_dir.exists():
        print(f"Embeddings not found: {emb_dir}")
        print(f"Run precompute for {args.dataset} first. See experiments/README.md")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg.get("num_classes") or 8

    # SData: zero-shot (trained head). Cross-dataset: linear-probe by default (measures transfer).
    use_linear_probe = args.linear_probe or (
        not args.zero_shot and args.dataset in ("video_mme", "nextqa", "charades", "egoschema", "vima_bench")
    )
    if use_linear_probe:
        print(f"Using linear-probe evaluation (train classifier on fused embeddings for {args.dataset})")
        results = run_evaluation_linear_probe(emb_dir, Path(args.checkpoint), out_dir, num_classes, device)
    else:
        results = run_evaluation(emb_dir, Path(args.checkpoint), out_dir, num_classes, device)
    results["dataset"] = cfg["name"]
    results["description"] = cfg.get("description", "")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # VisCoP-style table: L1 | L2 | L3 | A1..A8 | EgoSchema | NeXTQA | VideoMME | ADL-X | Avg
    dataset_key = args.dataset
    pc = results.get("per_class") or {}
    tax = cfg.get("taxonomy", "native")

    def _val(i, is_viscop=False):
        if is_viscop and tax == "viscop":
            v = pc.get(str(i), {}).get("accuracy")
            return f"{100*v:.1f}" if v is not None else "---"
        if tax == "sdata":
            v = pc.get(str(i), {}).get("accuracy")
            return f"{100*v:.1f}" if v is not None else "---"
        return "---"

    # CSV: VisCoP table format
    col_headers = ["Model", "L1", "L2", "L3"] + [f"A{i+1}" for i in range(8)] + ["EgoSchema", "NeXTQA", "VideoMME", "ADL-X", "Avg"]
    vals = ["MMFuse"]
    for i in range(3):
        vals.append(_val(i, is_viscop=True))
    for i in range(8):
        vals.append(_val(i))
    for k in ["egoschema", "nextqa", "video_mme", "charades"]:
        vals.append(f"{100*results['accuracy']:.1f}" if k == dataset_key and results.get("accuracy") is not None else "---")
    vals.append(f"{100*results['accuracy']:.1f}" if results.get("accuracy") is not None else "---")
    with open(out_dir / "results.csv", "w") as f:
        f.write(",".join(col_headers) + "\n")
        f.write(",".join(vals) + "\n")
        f.write("\n# A1=Start, A2=Focus, A3=Down, A4=Up, A5=Stop, A6=Left, A7=Right, A8=Perfect. L1/L2/L3=VIMA-Bench.\n")

    # HTML: VisCoP table format
    html_lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>MMFuse Results - " + cfg["name"] + "</title>",
        "<style>table{border-collapse:collapse;font-family:sans-serif}th,td{border:1px solid #333;padding:8px}th{background:#1a1a2e;color:#eee}.num{text-align:right}</style>",
        "</head><body><h1>MMFuse: " + cfg["name"] + "</h1>",
        "<h2>VisCoP-style: L1 | L2 | L3 | A1-A8 | Dataset-specific</h2>",
        "<table><tr><th>Model</th><th>L1</th><th>L2</th><th>L3</th>",
    ]
    for i in range(8):
        html_lines.append(f"<th>A{i+1}</th>")
    html_lines.append("<th>EgoSchema</th><th>NeXTQA</th><th>VideoMME</th><th>ADL-X</th><th>Avg</th></tr>")
    row = "<tr><td>MMFuse</td>" + "".join(f"<td class='num'>{v}</td>" for v in vals[1:]) + "</tr>"
    html_lines.append(row)
    html_lines.append("</table>")
    html_lines.append("<p><small><b>SData:</b> A1=Start, A2=Focus, A3=Down, A4=Up, A5=Stop, A6=Left, A7=Right, A8=Perfect. <b>VisCoP:</b> L1=Object Placement, L2=Novel Combination, L3=Novel Object.</small></p>")
    html_lines.append("</body></html>")
    with open(out_dir / "results.html", "w") as f:
        f.write("\n".join(html_lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        if results.get("per_class") and cfg.get("class_names"):
            n = len(cfg["class_names"])
            names = [f"A{i+1}" for i in range(n)]
            accs = [100 * results["per_class"].get(str(i), {}).get("accuracy", 0) for i in range(n)]
            ax.barh(names, accs, color=plt.cm.viridis([a/100 for a in accs]))
            ax.set_xlabel("Accuracy (%)")
            ax.set_title(f"{cfg['name']} - A1-A{n} (VisCoP-style)")
            ax.set_xlim(0, 100)
        else:
            ax.barh([cfg["name"]], [100 * results["accuracy"]], color="steelblue")
            ax.set_xlabel("Accuracy (%)")
            ax.set_title(f"{cfg['name']} - Overall")
            ax.set_xlim(0, 100)
        fig.tight_layout()
        fig.savefig(out_dir / "results_accuracy.png", dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        pass

    print(f"\n{'='*100}")
    print(f"RESULTS: {cfg['name']} (VisCoP-style table)")
    print(f"{'='*100}")
    print("Model   L1    L2    L3   " + "  ".join([f"A{i+1}" for i in range(8)]) + "  EgoSchema NeXTQA VideoMME ADL-X   Avg")
    print("-" * 100)
    print("  ".join(vals))
    print(f"{'='*100}")
    print("SData: A1=Start, A2=Focus, A3=Down, A4=Up, A5=Stop, A6=Left, A7=Right, A8=Perfect")
    print("VisCoP: L1=Object Placement, L2=Novel Combination, L3=Novel Object")
    print(f"Saved: {out_dir}/ (results.json, results.csv, results.html, results_accuracy.png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
