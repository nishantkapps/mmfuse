#!/usr/bin/env python3
"""
Run paper-ready comparison: MMFuse vs CLIP vs VisCoP on established datasets and SData.

Established datasets (NeXTQA, VideoMME, ADL-X): linear probe on embeddings.
SData: full comparison including CLIP vision-only.

Requires:
- embeddings/video_mme, embeddings/nextqa, embeddings/charades (VisCoP)
- embeddings/video_mme_clip, embeddings/nextqa_clip, embeddings/charades_clip (CLIP) - run precompute with --vision-encoder clip --text-encoder clip
- embeddings/sdata_viscop (VisCoP), embeddings/sdata_clip (CLIP vision) - for SData
- experiments/results/*/results.json (MMFuse from run_dataset --linear-probe)
- Checkpoint for MMFuse SData evaluation

Output: paper_comparison_established.csv/.tex, paper_comparison_sdata.csv/.tex, paper_comparison.tex (combined)

Usage:
  python experiments/run_paper_comparisons.py --checkpoint checkpoints/ckpt_sdata_epoch_N.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))


def run_linear_probe(emb_dir: Path, test_size: float = 0.2, random_state: int = 42):
    """Linear probe on embeddings. Returns dict with accuracy, precision, recall (macro). Uses vision+audio concat."""
    samples = sorted(emb_dir.glob("*.pt"))
    if not samples:
        return None
    v1, v2, a, y = [], [], [], []
    for p in samples:
        d = __import__("torch").load(p, map_location="cpu", weights_only=True)
        def to_np(t):
            return t.detach().numpy() if hasattr(t, "detach") else (t if isinstance(t, np.ndarray) else np.array(t))
        v1.append(to_np(d["vision_camera1"]))
        v2.append(to_np(d["vision_camera2"]))
        a.append(to_np(d["audio"]))
        y.append(int(d["target"].item() if hasattr(d["target"], "item") else d["target"]))
    v1, v2, a, y = np.stack(v1), np.stack(v2), np.stack(a), np.array(y)
    X = np.concatenate([v1, v2, a], axis=1).astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        tr_idx, te_idx = train_test_split(range(len(y)), test_size=test_size, stratify=y, random_state=random_state)
    except ValueError:
        tr_idx, te_idx = train_test_split(range(len(y)), test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_te = scaler.transform(X[te_idx])
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    clf = LogisticRegression(max_iter=1000, random_state=random_state, C=1.0)
    clf.fit(X_tr, y[tr_idx])
    y_pred = clf.predict(X_te)
    y_te = y[te_idx]
    acc = float(np.mean(y_pred == y_te))
    labels = np.unique(y)
    prec, rec, f1 = None, None, None
    if len(labels) <= 20:
        report = classification_report(y_te, y_pred, labels=labels, output_dict=True, zero_division=0)
        macro = report.get("macro avg", {})
        prec, rec = macro.get("precision"), macro.get("recall")
        f1 = macro.get("f1-score")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def load_mmfuse_result(results_dir: Path, dataset: str) -> dict | None:
    """Load MMFuse metrics from experiments/results/<dataset>/results.json or results.csv.
    Returns dict with accuracy, precision, recall (or None)."""
    ds_dir = results_dir / dataset
    # Prefer results.json (from run_dataset.py)
    p = ds_dir / "results.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        acc = d.get("accuracy")
        if acc is not None:
            pc = d.get("per_class") or {}
            prec, rec, f1 = None, None, None
            if pc:
                precs = [pc[k]["precision"] for k in pc if "precision" in pc.get(k, {})]
                recs = [pc[k]["recall"] for k in pc if "recall" in pc.get(k, {})]
                f1s = [pc[k]["f1-score"] for k in pc if "f1-score" in pc.get(k, {})]
                prec = float(np.mean(precs)) if precs else None
                rec = float(np.mean(recs)) if recs else None
                f1 = float(np.mean(f1s)) if f1s else None
            return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    # Fallback: parse results.csv (column VideoMME/NeXTQA/ADL-X) - accuracy only
    col_map = {"video_mme": "VideoMME", "nextqa": "NeXTQA", "charades": "ADL-X"}
    csv_path = ds_dir / "results.csv"
    if csv_path.exists():
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if row:
                v = row.get(col_map.get(dataset, dataset), "").strip()
                if v and v != "---":
                    try:
                        acc = float(v) / 100.0
                        return {"accuracy": acc, "precision": None, "recall": None, "f1": None}
                    except ValueError:
                        pass
    return None


def run_sdata_comparison(emb_viscop: Path, emb_clip: Path, ckpt_path: Path | None, device=None) -> dict:
    """Run SData comparison: CLIP, VisCoP vision-only, MMFuse. Returns dict of accuracies."""
    from experiments.run_sdata_baselines import (
        load_embeddings,
        run_baseline,
        run_mmfuse_eval,
    )

    out = {}
    if not emb_viscop.exists():
        return out

    v1, v2, a, y = load_embeddings(emb_viscop)
    tr_idx, te_idx = train_test_split(range(len(y)), test_size=0.2, stratify=y, random_state=42)

    vision = np.concatenate([v1, v2], axis=1)
    audio = a
    vision_audio = np.concatenate([vision, audio], axis=1)

    out["VisCoP vision-only"] = run_baseline(
        vision[tr_idx], y[tr_idx], vision[te_idx], y[te_idx], "v"
    )
    out["Audio-only (Wav2Vec)"] = run_baseline(
        audio[tr_idx], y[tr_idx], audio[te_idx], y[te_idx], "a"
    )
    out["VisCoP+Audio (concat)"] = run_baseline(
        vision_audio[tr_idx], y[tr_idx], vision_audio[te_idx], y[te_idx], "va"
    )

    if emb_clip.exists():
        v1_c, v2_c, a_c, y_c = load_embeddings(emb_clip)
        if len(y_c) == len(y):  # same sample order
            vision_c = np.concatenate([v1_c, v2_c], axis=1)
            out["CLIP vision-only"] = run_baseline(
                vision_c[tr_idx], y[tr_idx], vision_c[te_idx], y[te_idx], "c"
            )

    if ckpt_path and Path(ckpt_path).exists():
        import torch
        dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(dev, str):
            dev = torch.device(dev)
        res = run_mmfuse_eval(emb_viscop, Path(ckpt_path), tr_idx, te_idx, dev)
        out["MMFuse (full)"] = res
        out["_movement_mse"] = res.get("movement_mse")

    return out


def find_checkpoint(proj: Path) -> Path | None:
    """Find latest MMFuse checkpoint."""
    for pattern in ["checkpoints/ckpt_sdata_epoch_*.pt", "runs/*/ckpt_sdata_epoch_*.pt"]:
        candidates = sorted(proj.glob(pattern))
        if candidates:
            return candidates[-1]
    if (proj / "models/sdata_viscop/pytorch_model.bin").exists():
        return proj / "models/sdata_viscop/pytorch_model.bin"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="MMFuse checkpoint for SData (auto-find if missing)")
    p.add_argument("--device", default=None, help="Device for MMFuse SData eval: cuda, cuda:0, cpu. Default: cuda:0 if available. Use CUDA_VISIBLE_DEVICES=0 to restrict to GPU 0.")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--embeddings-dir", default=None)
    args = p.parse_args()

    proj = Path(__file__).resolve().parent.parent
    script_dir = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir) if args.results_dir else proj / "experiments" / "results"
    emb_root = Path(args.embeddings_dir) if args.embeddings_dir else proj / "embeddings"
    out_dir = results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint: use given path or auto-find
    ckpt_path = None
    if args.checkpoint and "N.pt" not in str(args.checkpoint):
        p0 = Path(args.checkpoint)
        if p0.is_absolute():
            ckpt_path = p0 if p0.exists() else None
        else:
            for base in [Path.cwd(), proj]:
                cand = (base / args.checkpoint).resolve()
                if cand.exists():
                    ckpt_path = cand
                    break
    if ckpt_path is None:
        ckpt_path = find_checkpoint(proj)
    args.checkpoint = str(ckpt_path) if ckpt_path else None

    # Resolve device for MMFuse SData eval
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "GPU"
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s), {gpu_name}")
    else:
        print("CUDA not available (PyTorch CPU build or no GPU). MMFuse SData eval will use CPU.")
    if args.device:
        if "cuda" in args.device.lower() and not cuda_available:
            print("WARNING: --device cuda requested but CUDA not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if cuda_available else "cpu")
    if args.checkpoint:
        print(f"MMFuse SData eval will use device: {device}")

    # Run collect to ensure results_table.csv is up to date (don't fail if it errors)
    import subprocess
    r = subprocess.run(
        [sys.executable, str(script_dir / "collect_cross_dataset_results.py")],
        cwd=str(proj),
        capture_output=True,
    )
    if r.returncode != 0 and r.stderr:
        print(f"Note: collect_cross_dataset_results had warnings: {r.stderr.decode()[:200]}")

    # Published numbers (cite from papers)
    PUBLISHED = {
        "VisCoP (paper)": {"nextqa": 83.71, "video_mme": 63.67, "charades": 61.26, "sdata": None, "note": "Cited from VisCoP paper"},
        "LLaVA-NeXT (paper)": {"nextqa": 77.31, "video_mme": 60.2, "charades": None, "sdata": None, "note": "Cited from LLaVA-NeXT"},
    }

    # Established datasets
    datasets = ["nextqa", "video_mme", "charades"]
    emb_suffixes = {"viscop": "", "clip": "_clip"}

    results = {}
    results_precision = {}
    results_recall = {}
    results_f1 = {}
    for model_name, row in PUBLISHED.items():
        results[model_name] = row.copy()
        results_precision[model_name] = {d: None for d in datasets + ["sdata"]}
        results_recall[model_name] = {d: None for d in datasets + ["sdata"]}
        results_f1[model_name] = {d: None for d in datasets + ["sdata"]}

    def _store_metrics(model, ds, res):
        results[model][ds] = res["accuracy"] * 100
        results_precision[model][ds] = res["precision"] * 100 if res.get("precision") is not None else None
        results_recall[model][ds] = res["recall"] * 100 if res.get("recall") is not None else None
        results_f1[model][ds] = res["f1"] * 100 if res.get("f1") is not None else None

    # VisCoP (linear probe) - our embeddings
    for ds in datasets:
        emb_dir = emb_root / (ds + emb_suffixes["viscop"])
        if emb_dir.exists():
            res = run_linear_probe(emb_dir)
            if res is not None:
                if "VisCoP (linear probe)" not in results:
                    results["VisCoP (linear probe)"] = {d: None for d in datasets + ["sdata"]}
                    results_precision["VisCoP (linear probe)"] = {d: None for d in datasets + ["sdata"]}
                    results_recall["VisCoP (linear probe)"] = {d: None for d in datasets + ["sdata"]}
                    results_f1["VisCoP (linear probe)"] = {d: None for d in datasets + ["sdata"]}
                _store_metrics("VisCoP (linear probe)", ds, res)

    # CLIP (linear probe)
    for ds in datasets:
        emb_dir = emb_root / (ds + emb_suffixes["clip"])
        if emb_dir.exists():
            res = run_linear_probe(emb_dir)
            if res is not None:
                if "CLIP (linear probe)" not in results:
                    results["CLIP (linear probe)"] = {d: None for d in datasets + ["sdata"]}
                    results_precision["CLIP (linear probe)"] = {d: None for d in datasets + ["sdata"]}
                    results_recall["CLIP (linear probe)"] = {d: None for d in datasets + ["sdata"]}
                    results_f1["CLIP (linear probe)"] = {d: None for d in datasets + ["sdata"]}
                _store_metrics("CLIP (linear probe)", ds, res)

    # Linear probe on SData (VisCoP, CLIP)
    for name, emb_subdir in [("VisCoP (linear probe)", "sdata_viscop"), ("CLIP (linear probe)", "sdata_clip")]:
        emb_dir = emb_root / emb_subdir
        if emb_dir.exists():
            res = run_linear_probe(emb_dir)
            if res is not None:
                if name not in results:
                    results[name] = {d: None for d in datasets + ["sdata"]}
                    results_precision[name] = {d: None for d in datasets + ["sdata"]}
                    results_recall[name] = {d: None for d in datasets + ["sdata"]}
                    results_f1[name] = {d: None for d in datasets + ["sdata"]}
                _store_metrics(name, "sdata", res)

    # MMFuse: read individual runs from each dataset folder
    mmfuse_row = {d: None for d in datasets + ["sdata"]}
    mmfuse_precision = {d: None for d in datasets + ["sdata"]}
    mmfuse_recall = {d: None for d in datasets + ["sdata"]}
    mmfuse_f1 = {d: None for d in datasets + ["sdata"]}
    for ds in datasets:
        res = load_mmfuse_result(results_dir, ds)
        if res is not None:
            mmfuse_row[ds] = res["accuracy"] * 100
            mmfuse_precision[ds] = res["precision"] * 100 if res.get("precision") is not None else None
            mmfuse_recall[ds] = res["recall"] * 100 if res.get("recall") is not None else None
            mmfuse_f1[ds] = res["f1"] * 100 if res.get("f1") is not None else None

    # SData comparison
    sdata_results = run_sdata_comparison(
        emb_root / "sdata_viscop",
        emb_root / "sdata_clip",
        args.checkpoint,
        device=device,
    )
    movement_mse = sdata_results.pop("_movement_mse", None)
    for model_name, res in sdata_results.items():
        if model_name not in results:
            results[model_name] = {d: None for d in datasets + ["sdata"]}
            results_precision[model_name] = {d: None for d in datasets + ["sdata"]}
            results_recall[model_name] = {d: None for d in datasets + ["sdata"]}
            results_f1[model_name] = {d: None for d in datasets + ["sdata"]}
        acc = res["accuracy"] if isinstance(res, dict) else res
        results[model_name]["sdata"] = acc * 100
        if isinstance(res, dict):
            results_precision[model_name]["sdata"] = res["precision"] * 100 if res.get("precision") is not None else None
            results_recall[model_name]["sdata"] = res["recall"] * 100 if res.get("recall") is not None else None
            results_f1[model_name]["sdata"] = res["f1"] * 100 if res.get("f1") is not None else None
    if "MMFuse (full)" in sdata_results:
        res = sdata_results["MMFuse (full)"]
        mmfuse_row["sdata"] = (res["accuracy"] if isinstance(res, dict) else res) * 100
        if isinstance(res, dict):
            mmfuse_precision["sdata"] = res["precision"] * 100 if res.get("precision") is not None else None
            mmfuse_recall["sdata"] = res["recall"] * 100 if res.get("recall") is not None else None
            mmfuse_f1["sdata"] = res["f1"] * 100 if res.get("f1") is not None else None
    elif "MMFuse (full)" in results:
        mmfuse_row["sdata"] = results["MMFuse (full)"]["sdata"]
    results["MMFuse"] = mmfuse_row
    results_precision["MMFuse"] = mmfuse_precision
    results_recall["MMFuse"] = mmfuse_recall
    results_f1["MMFuse"] = mmfuse_f1

    # Two-table layout: Table 1 = established benchmarks, Table 2 = SData
    models_established = [
        "VisCoP (paper)",
        "LLaVA-NeXT (paper)",
        "CLIP (linear probe)",
        "VisCoP (linear probe)",
        "MMFuse",
    ]
    models_sdata = [
        "MMFuse",
        "VisCoP (linear probe)",
        "CLIP (linear probe)",
        "VisCoP vision-only",
        "Audio-only (Wav2Vec)",
        "VisCoP+Audio (concat)",
        "CLIP vision-only",
    ]
    models_established = [m for m in models_established if m in results]
    models_sdata = [m for m in models_sdata if m in results]

    def fmt(v):
        if v is None:
            return "---"
        return f"{v:.1f}"

    # --- Table 1: Established benchmarks (NeXTQA, VideoMME, ADL-X) - Acc, Prec, Rec, F1 per dataset ---
    metric_cols = ["Acc", "Prec", "Rec", "F1"]
    csv_hdr_est = "model," + ",".join(f"NeXTQA_{m}" for m in metric_cols) + "," + ",".join(f"VideoMME_{m}" for m in metric_cols) + "," + ",".join(f"ADL-X_{m}" for m in metric_cols)
    csv_est = out_dir / "paper_comparison_established.csv"
    with open(csv_est, "w") as f:
        f.write(csv_hdr_est + "\n")
        for m in models_established:
            r, rp, rr, rf = results[m], results_precision.get(m, {}), results_recall.get(m, {}), results_f1.get(m, {})
            cells = []
            for ds, lbl in [("nextqa", "NeXTQA"), ("video_mme", "VideoMME"), ("charades", "ADL-X")]:
                cells.extend([fmt(r.get(ds)), fmt(rp.get(ds)), fmt(rr.get(ds)), fmt(rf.get(ds))])
            f.write(f'"{m}",' + ",".join(cells) + "\n")
    print(f"Saved {csv_est}")

    tex_cols_est = " & ".join([f"NeXTQA {m}" for m in metric_cols] + [f"VideoMME {m}" for m in metric_cols] + [f"ADL-X {m}" for m in metric_cols])
    tex_est = out_dir / "paper_comparison_established.tex"
    lines_est = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison on established benchmarks (NeXTQA, VideoMME, ADL-X). Acc=Accuracy, Prec=Precision, Rec=Recall, F1=F1-score (macro \%). Published numbers: accuracy only.}",
        r"\label{tab:paper_comparison_established}",
        r"\begin{tabular}{l" + "cccc" * 3 + "}",
        r"\toprule",
        r"Model & " + tex_cols_est + r" \\",
        r"\midrule",
    ]
    for m in models_established:
        r, rp, rr, rf = results[m], results_precision.get(m, {}), results_recall.get(m, {}), results_f1.get(m, {})
        m_esc = m.replace("_", r"\_")
        cells = []
        for ds in ["nextqa", "video_mme", "charades"]:
            cells.extend([fmt(r.get(ds)), fmt(rp.get(ds)), fmt(rr.get(ds)), fmt(rf.get(ds))])
        row = " & ".join(cells) + r" \\"
        if m == "MMFuse":
            lines_est.append(rf"\textbf{{{m_esc}}} & " + row)
        else:
            lines_est.append(f"{m_esc} & " + row)
    lines_est.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(tex_est, "w") as f:
        f.write("\n".join(lines_est))
    print(f"Saved {tex_est}")

    # --- Table 2: SData - Acc, Prec, Rec, F1, Mov.MSE ---
    csv_hdr_sdata = "model,SData_Acc,SData_Prec,SData_Rec,SData_F1,Movement_MSE"
    csv_sdata = out_dir / "paper_comparison_sdata.csv"
    with open(csv_sdata, "w") as f:
        f.write(csv_hdr_sdata + "\n")
        for m in models_sdata:
            r, rp, rr, rf = results[m], results_precision.get(m, {}), results_recall.get(m, {}), results_f1.get(m, {})
            mse_str = f"{movement_mse:.4f}" if m == "MMFuse" and movement_mse is not None else "---"
            f.write(f'"{m}",{fmt(r.get("sdata"))},{fmt(rp.get("sdata"))},{fmt(rr.get("sdata"))},{fmt(rf.get("sdata"))},{mse_str}\n')
    print(f"Saved {csv_sdata}")

    tex_sdata = out_dir / "paper_comparison_sdata.tex"
    lines_sdata = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison on SData. Acc=Accuracy, Prec=Precision, Rec=Recall, F1=F1-score (macro \%). MMFuse uses trained fusion; others use linear probe or single-modality baselines.}",
        r"\label{tab:paper_comparison_sdata}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Acc & Prec & Rec & F1 & Mov. MSE \\",
        r"\midrule",
    ]
    for m in models_sdata:
        r, rp, rr, rf = results[m], results_precision.get(m, {}), results_recall.get(m, {}), results_f1.get(m, {})
        m_esc = m.replace("_", r"\_")
        mse_str = f"{movement_mse:.4f}" if m == "MMFuse" and movement_mse is not None else "---"
        row = f"{fmt(r.get('sdata'))} & {fmt(rp.get('sdata'))} & {fmt(rr.get('sdata'))} & {fmt(rf.get('sdata'))} & {mse_str}"
        if m == "MMFuse":
            lines_sdata.append(rf"\textbf{{{m_esc}}} & " + row + r" \\")
        else:
            lines_sdata.append(f"{m_esc} & " + row + r" \\")
    lines_sdata.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(tex_sdata, "w") as f:
        f.write("\n".join(lines_sdata))
    print(f"Saved {tex_sdata}")

    # Combined LaTeX (both tables)
    tex_path = out_dir / "paper_comparison.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines_est) + "\n\n" + "\n".join(lines_sdata))
    print(f"Saved {tex_path}")

    # Console output
    print("\n" + "=" * 100)
    print("Table 1: Established Benchmarks (NeXTQA, VideoMME, ADL-X) - Acc | Prec | Rec | F1")
    print("=" * 100)
    print(f"{'Model':<26}  NeXTQA (Acc|Prec|Rec|F1)   VideoMME (Acc|Prec|Rec|F1)   ADL-X (Acc|Prec|Rec|F1)")
    print("-" * 100)
    for m in models_established:
        r, rp, rr, rf = results[m], results_precision.get(m, {}), results_recall.get(m, {}), results_f1.get(m, {})
        parts = []
        for ds in ["nextqa", "video_mme", "charades"]:
            parts.append(f"{fmt(r.get(ds))}|{fmt(rp.get(ds))}|{fmt(rr.get(ds))}|{fmt(rf.get(ds))}")
        print(f"{m:<26}  {parts[0]:<24} {parts[1]:<24} {parts[2]}")
    print("=" * 100)
    print("\nTable 2: SData - Acc | Prec | Rec | F1 | Mov.MSE")
    print("=" * 70)
    print(f"{'Model':<28} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Mov.MSE'}")
    print("-" * 70)
    for m in models_sdata:
        r, rp, rr, rf = results[m], results_precision.get(m, {}), results_recall.get(m, {}), results_f1.get(m, {})
        mse_str = f"{movement_mse:.4f}" if m == "MMFuse" and movement_mse else "---"
        print(f"{m:<28} {fmt(r.get('sdata')):<8} {fmt(rp.get('sdata')):<8} {fmt(rr.get('sdata')):<8} {fmt(rf.get('sdata')):<8} {mse_str}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
