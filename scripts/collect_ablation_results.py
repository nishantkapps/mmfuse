#!/usr/bin/env python3
"""
Collect ablation results and generate paper-ready table and figures.
Run after completing ablation experiments with scripts/run_ablation.py.

Output:
  - Table: Experiment | ACC (%) | Precision (%) | Recall (%) | Δ from Full (↑)
    (Precision/Recall columns shown when present in ablation_results.json)
  - Figure 1: Bar chart of test accuracy per experiment
  - Figure 2: Bar chart of Δ from full (impact of each ablation)
  - Figure 3: Scatter plot of Δtrain vs Δtest (relative to full model)

Usage:
  python scripts/collect_ablation_results.py --ablation-dir ablation_runs --output-dir results

  # Recompute precision/recall from existing checkpoints (no re-training):
  python scripts/collect_ablation_results.py --recompute-metrics --embeddings-dir embeddings/sdata_viscop --dataset dataset/sdata
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader, Subset

_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))


def _safe_float(val, default=None):
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    return default


def _recompute_metrics_for_ablation(
    run_dir: Path,
    embeddings_dir: Path,
    dataset_path: Optional[Path],
    device: str = "cuda",
) -> Optional[Tuple[float, float, float]]:
    """Run inference on test set and return (accuracy, precision, recall). Returns None on error."""
    results_path = run_dir / "ablation_results.json"
    config_path = run_dir / "ablation_config.json"
    if not results_path.exists() or not config_path.exists():
        return None
    ckpts = sorted(run_dir.glob("ckpt_sdata_epoch_*.pt"))
    if not ckpts:
        return None

    with open(config_path) as f:
        ablation_cfg = json.load(f)
    with open(embeddings_dir / "config.json") as f:
        emb_config = json.load(f)

    vision_dim = emb_config["vision_dim"]
    audio_dim = emb_config.get("audio_dim", 768)
    num_classes = emb_config.get("num_classes", 8)
    cross_pair = emb_config.get("cross_pair", False)
    augment_variations = emb_config.get("augment_variations", 16)

    try:
        from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
        from mmfuse.fusion.multimodal_fusion import MultimodalFusion, MultimodalFusionWithAttention
    except ModuleNotFoundError:
        from encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
        from fusion.multimodal_fusion import MultimodalFusion, MultimodalFusionWithAttention

    _run_ablation = Path(__file__).resolve().parent / "run_ablation.py"
    spec = importlib.util.spec_from_file_location("run_ablation", _run_ablation)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    PrecomputedSDataDataset = mod.PrecomputedSDataDataset
    build_embedding_ablation = mod.build_embedding_ablation
    ActionClassifier = mod.ActionClassifier
    collate_fn = mod.collate_fn

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ds = PrecomputedSDataDataset(embeddings_dir, emb_config)

    if dataset_path and dataset_path.exists():
        _, test_idx = mod._get_train_test_indices_split_before_aug(
            dataset_path, cross_pair, augment_variations, len(ds.samples)
        )
    else:
        from sklearn.model_selection import train_test_split
        labels = [torch.load(p, map_location="cpu", weights_only=True)["target"] for p in ds.samples]
        try:
            _, test_idx = train_test_split(range(len(ds)), test_size=0.1, stratify=labels, random_state=42)
        except ValueError:
            _, test_idx = train_test_split(range(len(ds)), test_size=0.1, random_state=42)

    test_ds = Subset(ds, test_idx)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    modality_dims = {
        "vision_camera1": vision_dim,
        "vision_camera2": vision_dim,
        "audio": audio_dim,
        "pressure": 256,
        "emg": 256,
    }
    pressure = PressureSensorEncoder(output_dim=256, input_features=2).to(dev)
    emg = EMGSensorEncoder(output_dim=256, num_channels=3, input_features=4).to(dev)

    ckpt = torch.load(ckpts[-1], map_location=dev, weights_only=True)
    fusion_dim = ckpt.get("fusion_dim", 256)
    fusion_type = ablation_cfg.get("fusion_type", "attention")
    if fusion_type == "concat":
        fusion = MultimodalFusion(
            modality_dims=modality_dims,
            fusion_dim=fusion_dim,
            fusion_method="concat_project",
            dropout=0.2,
        ).to(dev)
    else:
        fusion = MultimodalFusionWithAttention(
            modality_dims=modality_dims,
            fusion_dim=fusion_dim,
            num_heads=8,
            dropout=0.2,
        ).to(dev)
    model = ActionClassifier(embedding_dim=fusion_dim, num_classes=num_classes).to(dev)
    encoders = {"pressure": pressure, "emg": emg, "fusion": fusion}
    fusion.load_state_dict(ckpt["fusion_state"])
    model.load_state_dict(ckpt["model_state"])
    fusion.eval()
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_dl:
            targets = torch.tensor([s["target"] for s in batch], dtype=torch.long).to(dev)
            fused, _ = build_embedding_ablation(batch, dev, encoders, ablation_cfg)
            preds = model(fused).argmax(dim=1)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    return (acc, prec, rec)


def load_results(root: Path, configs: dict) -> List[Dict[str, Any]]:
    """Load all ablation results. Returns list of dicts with ablation, test_accuracy, precision, recall, etc."""
    rows = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        results_path = run_dir / "ablation_results.json"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            data = json.load(f)
        ablation_name = data.get("ablation", run_dir.name)
        desc = configs.get(ablation_name, {}).get("description", "")
        test_acc = _safe_float(data.get("test_accuracy", data.get("test_acc", 0)), 0.0)
        train_acc = _safe_float(data.get("train_accuracy", data.get("train_acc", 0)), 0.0)
        test_precision = _safe_float(data.get("test_precision", data.get("precision", data.get("precision_macro"))))
        test_recall = _safe_float(data.get("test_recall", data.get("recall", data.get("recall_macro"))))
        rows.append({
            "ablation": ablation_name,
            "description": desc,
            "test_accuracy": test_acc,
            "train_accuracy": train_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation-dir", default="ablation_runs", help="Root dir with ablation run folders")
    p.add_argument("--output-dir", default="results", help="Output directory for table and figures")
    p.add_argument("--config", default="config/ablation_experiments.yaml", help="Ablation config")
    p.add_argument("--format", choices=["all", "table", "figures"], default="all",
                   help="Generate: all (table+figures), table only, or figures only")
    p.add_argument("--recompute-metrics", action="store_true",
                   help="Recompute precision/recall from checkpoints (requires --embeddings-dir and --dataset)")
    p.add_argument("--embeddings-dir", default=None, help="Path to precomputed embeddings (for --recompute-metrics)")
    p.add_argument("--dataset", default=None, help="Path to sdata dataset (for --recompute-metrics, correct test split)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    root = Path(args.ablation_dir)
    if not root.exists():
        print(f"Ablation dir not found: {root}")
        return 1

    if args.recompute_metrics:
        if not args.embeddings_dir:
            print("--recompute-metrics requires --embeddings-dir")
            return 1
        emb_dir = Path(args.embeddings_dir)
        if not emb_dir.exists():
            print(f"Embeddings dir not found: {emb_dir}")
            return 1
        dataset_path = Path(args.dataset) if args.dataset else None
        print("Recomputing precision/recall from checkpoints...")
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            result = _recompute_metrics_for_ablation(
                run_dir, emb_dir, dataset_path, args.device
            )
            if result is None:
                print(f"  {run_dir.name}: skipped (no checkpoint or config)")
                continue
            acc, prec, rec = result
            results_path = run_dir / "ablation_results.json"
            with open(results_path) as f:
                data = json.load(f)
            data["test_accuracy"] = acc
            data["test_precision"] = prec
            data["test_recall"] = rec
            with open(results_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  {run_dir.name}: ACC={acc*100:.2f}% Prec={prec*100:.2f}% Rec={rec*100:.2f}%")
        print("Done. Updated ablation_results.json in each run dir.\n")

    config_path = Path(__file__).resolve().parent.parent / args.config
    configs = {}
    if config_path.exists():
        with open(config_path) as f:
            configs = yaml.safe_load(f)

    rows = load_results(root, configs)
    if not rows:
        print("No ablation results found.")
        return 1

    # Assign run numbers (ablation_run_1, 2, 3...) by config order
    ablation_order = list(configs.keys()) if configs else []
    ablation_to_run = {name: i + 1 for i, name in enumerate(ablation_order)}
    for r in rows:
        r["run_number"] = ablation_to_run.get(r["ablation"], 999)
    rows.sort(key=lambda x: x["run_number"])
    for r in rows:
        r["run_name"] = f"ablation_run_{r['run_number']}"

    # Get full model baseline
    full_row = next((r for r in rows if r["ablation"] == "full"), None)
    full_test = full_row["test_accuracy"] * 100 if full_row else 0.0
    full_train = full_row["train_accuracy"] * 100 if full_row else 0.0
    full_precision = full_row["test_precision"] if full_row and full_row["test_precision"] is not None else None
    full_recall = full_row["test_recall"] if full_row and full_row["test_recall"] is not None else None

    # Compute Δ from full (in percentage points)
    for r in rows:
        r["test_acc_pct"] = r["test_accuracy"] * 100
        r["train_acc_pct"] = r["train_accuracy"] * 100
        r["precision_pct"] = r["test_precision"] * 100 if r["test_precision"] is not None else None
        r["recall_pct"] = r["test_recall"] * 100 if r["test_recall"] is not None else None
        r["delta_test"] = (r["test_accuracy"] - (full_row["test_accuracy"] if full_row else 0)) * 100
        r["delta_train"] = (r["train_accuracy"] - (full_row["train_accuracy"] if full_row else 0)) * 100
        r["delta_precision"] = (r["test_precision"] - full_precision) * 100 if r["test_precision"] is not None and full_precision is not None else None
        r["delta_recall"] = (r["test_recall"] - full_recall) * 100 if r["test_recall"] is not None and full_recall is not None else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Table ---
    if args.format in ("all", "table"):
        has_prec_recall = any(r["test_precision"] is not None or r["test_recall"] is not None for r in rows)
        table_path = out_dir / "ablation_table.tex"
        if has_prec_recall:
            header = r"Experiment & ACC (\%) & Precision (\%) & Recall (\%) & $\Delta$ from Full ($\uparrow$) \\"
            lines = [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Ablation study on MMFuse SData action classification.}",
                r"\label{tab:ablation}",
                r"\begin{tabular}{lcccc}",
                r"\toprule",
                header,
                r"\midrule",
            ]
            for r in rows:
                name = r["run_name"].replace("_", r"\_")
                acc_str = f"{r['test_acc_pct']:.2f}"
                prec_str = f"{r['precision_pct']:.2f}" if r["precision_pct"] is not None else "---"
                rec_str = f"{r['recall_pct']:.2f}" if r["recall_pct"] is not None else "---"
                if r["ablation"] == "full":
                    delta_display = "---"
                    line = f"\\textbf{{{name}}} & {acc_str} & {prec_str} & {rec_str} & {delta_display} \\\\"
                else:
                    delta_str = f"+{r['delta_test']:.2f}" if r["delta_test"] >= 0 else f"{r['delta_test']:.2f}"
                    line = f"{name} & {acc_str} & {prec_str} & {rec_str} & {delta_str} \\\\"
                lines.append(line)
        else:
            lines = [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Ablation study on MMFuse SData action classification.}",
                r"\label{tab:ablation}",
                r"\begin{tabular}{lcc}",
                r"\toprule",
                r"Experiment & ACC (\%) & $\Delta$ from Full ($\uparrow$) \\",
                r"\midrule",
            ]
            for r in rows:
                name = r["run_name"].replace("_", r"\_")
                acc_str = f"{r['test_acc_pct']:.2f}"
                if r["ablation"] == "full":
                    delta_display = "---"
                    line = f"\\textbf{{{name}}} & {acc_str} & {delta_display} \\\\"
                else:
                    delta_str = f"+{r['delta_test']:.2f}" if r["delta_test"] >= 0 else f"{r['delta_test']:.2f}"
                    line = f"{name} & {acc_str} & {delta_str} \\\\"
                lines.append(line)
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        with open(table_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Table saved to {table_path}")

        # CSV
        csv_path = out_dir / "ablation_table.csv"
        if csv_path.exists() and csv_path.is_dir():
            csv_path = out_dir / "ablation_metrics.csv"
            print(f"Note: ablation_table.csv is a directory, writing to {csv_path.name} instead")
        with open(csv_path, "w") as f:
            if has_prec_recall:
                f.write("run,ablation,description,acc_pct,precision_pct,recall_pct,delta_from_full\n")
                for r in rows:
                    desc = r["description"].replace('"', '""')
                    prec = f"{r['precision_pct']:.2f}" if r["precision_pct"] is not None else ""
                    rec = f"{r['recall_pct']:.2f}" if r["recall_pct"] is not None else ""
                    delta_str = "" if r["ablation"] == "full" else f"{r['delta_test']:.2f}"
                    f.write(f'"{r["run_name"]}","{r["ablation"]}","{desc}",{r["test_acc_pct"]:.2f},{prec},{rec},{delta_str}\n')
            else:
                f.write("run,ablation,description,acc_pct,delta_from_full\n")
                for r in rows:
                    desc = r["description"].replace('"', '""')
                    delta_str = "" if r["ablation"] == "full" else f"{r['delta_test']:.2f}"
                    f.write(f'"{r["run_name"]}","{r["ablation"]}","{desc}",{r["test_acc_pct"]:.2f},{delta_str}\n')
        print(f"CSV saved to {csv_path}")

    # --- Figures ---
    if args.format in ("all", "figures"):
        # Order: full first, then by delta (most negative first for ablations)
        sorted_rows = sorted(rows, key=lambda x: (0 if x["ablation"] == "full" else 1, -x["delta_test"]))
        names = [r["run_name"].replace("_", " ") for r in sorted_rows]
        test_accs = [r["test_acc_pct"] for r in sorted_rows]
        deltas = [r["delta_test"] for r in sorted_rows]

        # Figure 1: Bar chart of test accuracy
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        colors = ["#2ecc71" if r["ablation"] == "full" else "#3498db" for r in sorted_rows]
        bars = ax1.barh(names, test_accs, color=colors)
        ax1.axvline(x=full_test, color="gray", linestyle="--", alpha=0.7, label="Full model")
        ax1.set_xlabel("Test Accuracy (%)")
        ax1.set_title("Ablation: Test Accuracy by Experiment")
        ax1.set_xlim(0, 105)
        ax1.legend()
        ax1.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig1_path = out_dir / "ablation_fig_accuracy.png"
        fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print(f"Figure 1 saved to {fig1_path}")

        # Figure 2: Bar chart of Δ from full
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        non_full = [r for r in sorted_rows if r["ablation"] != "full"]
        if non_full:
            names2 = [r["run_name"].replace("_", " ") for r in non_full]
            deltas2 = [r["delta_test"] for r in non_full]
            colors2 = ["#e74c3c" if d < 0 else "#2ecc71" for d in deltas2]
            ax2.barh(names2, deltas2, color=colors2)
            ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
            ax2.set_xlabel(r"$\Delta$ from Full (percentage points)")
            ax2.set_title(r"Ablation: Impact of Removing Components ($\Delta$ from Full Model)")
            ax2.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig2_path = out_dir / "ablation_fig_delta.png"
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Figure 2 saved to {fig2_path}")

        # Figure 3: Scatter Δtrain vs Δtest (relative to full)
        non_full_rows = [r for r in rows if r["ablation"] != "full"]
        if len(non_full_rows) >= 2:
            fig3, ax3 = plt.subplots(figsize=(8, 8))
            delta_train = [r["delta_train"] for r in non_full_rows]
            delta_test = [r["delta_test"] for r in non_full_rows]
            labels = [r["run_name"].replace("_", " ") for r in non_full_rows]
            scatter = ax3.scatter(delta_train, delta_test, s=100, alpha=0.8, c=range(len(non_full_rows)), cmap="viridis")
            for i, lbl in enumerate(labels):
                ax3.annotate(lbl, (delta_train[i], delta_test[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
            ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax3.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
            ax3.set_xlabel(r"$\Delta$ train (relative to full model)")
            ax3.set_ylabel(r"$\Delta$ test (relative to full model)")
            ax3.set_title("Ablation: Train vs Test Accuracy Change")
            ax3.grid(alpha=0.3)
            plt.tight_layout()
            fig3_path = out_dir / "ablation_fig_scatter.png"
            fig3.savefig(fig3_path, dpi=150, bbox_inches="tight")
            plt.close(fig3)
            print(f"Figure 3 saved to {fig3_path}")

    print("\n--- Preview ---")
    for r in rows:
        delta_str = f" (Δ={r['delta_test']:+.2f})" if r["ablation"] != "full" else " [full]"
        parts = [f"ACC={r['test_acc_pct']:.2f}%"]
        if r["precision_pct"] is not None:
            parts.append(f"Prec={r['precision_pct']:.2f}%")
        if r["recall_pct"] is not None:
            parts.append(f"Rec={r['recall_pct']:.2f}%")
        print(f"  {r['run_name']}: {', '.join(parts)}{delta_str}")
    return 0


if __name__ == "__main__":
    exit(main())
