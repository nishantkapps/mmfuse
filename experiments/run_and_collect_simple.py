#!/usr/bin/env python3
"""
Run MMFuse evaluation on key datasets and collect a simple, paper-ready summary table.

Datasets covered (by default):
  - sdata
  - nextqa
  - video_mme
  - charades

This script:
  1) Calls experiments/run_dataset.py for each dataset with the given checkpoint.
  2) Reads experiments/results/<dataset>/results.json.
  3) Writes a compact CSV summary:
       experiments/simple_results.csv
     with columns:
       dataset,accuracy,num_samples,precision_macro,recall_macro,f1_macro,movement_mse

Usage (from repo root):

  python experiments/run_and_collect_simple.py \\
    --checkpoint mmfuse/checkpoints_clip_wav2vec_v3/model_v3.pt
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    proj_root = Path(__file__).resolve().parent.parent
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pt) relative to repo root or absolute",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["sdata", "nextqa", "video_mme", "charades"],
        help="Datasets to evaluate (default: sdata nextqa video_mme charades)",
    )
    p.add_argument(
        "--results-out",
        default=str(proj_root / "experiments" / "simple_results.csv"),
        help="Output CSV path (default: experiments/simple_results.csv)",
    )
    args = p.parse_args()

    checkpoint = args.checkpoint
    if not Path(checkpoint).is_absolute():
        checkpoint_path = proj_root / checkpoint
    else:
        checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return 1

    script_dir = proj_root / "experiments"
    results_root = script_dir / "results"

    # 1) Run evaluations
    for name in args.datasets:
        print("\n" + "=" * 60)
        print(f"Running evaluation for dataset: {name}")
        print("=" * 60)
        ret = subprocess.run(
            [
                sys.executable,
                str(script_dir / "run_dataset.py"),
                "--dataset",
                name,
                "--checkpoint",
                str(checkpoint_path),
            ],
            cwd=str(proj_root),
        )
        if ret.returncode != 0:
            print(f"[FAIL] {name} exited with code {ret.returncode}")
        else:
            print(f"[OK] {name}")

    # 2) Collect results
    rows = []
    for name in args.datasets:
        res_file = results_root / name / "results.json"
        if not res_file.exists():
            print(f"[WARN] No results.json for {name} at {res_file}")
            continue
        with open(res_file) as f:
            data = json.load(f)
        rows.append(
            {
                "dataset": data.get("dataset", name),
                "accuracy": data.get("accuracy"),
                "num_samples": data.get("num_samples"),
                "precision_macro": data.get("precision_macro"),
                "recall_macro": data.get("recall_macro"),
                "f1_macro": data.get("f1_macro"),
                "movement_mse": data.get("movement_mse"),
            }
        )

    out_path = Path(args.results_out)
    if not out_path.is_absolute():
        out_path = proj_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(
            "dataset,accuracy,num_samples,precision_macro,recall_macro,f1_macro,movement_mse\n"
        )
        for r in rows:
            acc = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else ""
            n = str(r["num_samples"]) if r["num_samples"] is not None else ""
            prec = f"{r['precision_macro']:.4f}" if r["precision_macro"] is not None else ""
            rec = f"{r['recall_macro']:.4f}" if r["recall_macro"] is not None else ""
            f1 = f"{r['f1_macro']:.4f}" if r["f1_macro"] is not None else ""
            mse = f"{r['movement_mse']:.4f}" if r["movement_mse"] is not None else ""
            name = str(r["dataset"]).replace(",", " ")
            f.write(f"{name},{acc},{n},{prec},{rec},{f1},{mse}\n")

    print(f"\nSimple summary CSV written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

