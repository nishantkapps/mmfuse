#!/usr/bin/env python3
"""
Evaluate MMFuse action classification on a manifest (test split by default).
Modes: full (vision + audio) | vision_only (zero audio, matches predict_video_only).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_predict_module():
    path = REPO_ROOT / "scripts" / "predict_with_model.py"
    spec = importlib.util.spec_from_file_location("predict_with_model", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            row["label"] = int(row["label"])
            row["aug_v"] = int(row["aug_v"])
            row["sample_id"] = int(row["sample_id"])
            rows.append(row)
    return rows


def _summarize():
    try:
        from sdata_vla_benchmark.metrics.classification import summarize
    except ImportError:
        import importlib.util

        mpath = REPO_ROOT / "sdata_vla_benchmark" / "metrics" / "classification.py"
        spec = importlib.util.spec_from_file_location("sdata_metrics", mpath)
        mmod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mmod)
        summarize = mmod.summarize
    return summarize


def evaluate_mmfuse(
    manifest: Path,
    checkpoint: Path,
    mode: str,
    split: str,
    output: Path | None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run MMFuse on manifest rows; optionally write JSON. Used by run_frozen_benchmarks."""
    pm = _load_predict_module()
    dev = device or pm.setup_environment()
    encoders = pm.load_encoders(dev)
    head, labels = pm.load_checkpoint(dev, encoders, str(checkpoint))

    rows = load_manifest(manifest)
    if split != "all":
        rows = [r for r in rows if r["split"] == split]

    predictions = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for row in rows:
        c1, c2, audio = row["cam1"], row["cam2"], row["audio"] or None
        gt = row["label"]
        if mode == "vision_only":
            pred_label, conf = pm.predict_video_only(encoders, head, c1, c2, dev, labels)
        else:
            pred_label, conf, _ = pm.predict_sample(
                encoders, head, c1, c2, audio if audio else None, dev, labels
            )

        try:
            pred_idx = labels.index(pred_label)
        except ValueError:
            pred_idx = -1
        y_true.append(gt)
        y_pred.append(pred_idx if pred_idx >= 0 else 0)

        predictions.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "label": gt,
                "pred": pred_label,
                "pred_idx": pred_idx,
                "confidence": conf,
                "cam1": c1,
                "cam2": c2,
                "mode": mode,
            }
        )

    summarize = _summarize()
    metrics = summarize(y_true, y_pred, num_classes=len(labels))

    out: dict[str, Any] = {
        "model": "mmfuse",
        "checkpoint": str(checkpoint),
        "mode": mode,
        "split": split,
        "manifest": str(manifest),
        "metrics": metrics,
        "predictions": predictions,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser(description="MMFuse eval on manifest (action accuracy).")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--mode", choices=("full", "vision_only"), default="full")
    ap.add_argument("--split", default="test", help="train | test | all")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out = evaluate_mmfuse(
        args.manifest,
        args.checkpoint,
        args.mode,
        args.split,
        args.output,
        device=args.device,
    )
    print(json.dumps({"wrote": str(args.output), **out["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
