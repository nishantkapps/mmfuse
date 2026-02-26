#!/usr/bin/env python3
"""
Run MMFuse evaluation on all cross-dataset benchmarks.
Skips datasets that don't have precomputed embeddings.
Auto-picks latest checkpoint from training if --checkpoint not specified.

Usage:
  python experiments/run_all_experiments.py
  python experiments/run_all_experiments.py --checkpoint path/to/model.pt
  python experiments/run_all_experiments.py --datasets sdata charades
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    p = argparse.ArgumentParser()
    proj = Path(__file__).resolve().parent.parent
    p.add_argument("--checkpoint", default=str(proj / "checkpoints/model.pt"), help="Path to model file (default: checkpoints/model.pt)")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Specific datasets to run (default: all)")
    p.add_argument("--skip-missing", action="store_true", default=True,
                   help="Skip datasets without embeddings (default: True)")
    args = p.parse_args()

    checkpoint = args.checkpoint

    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config" / "datasets.yaml"
    with open(config_path) as f:
        configs = yaml.safe_load(f)

    proj_root = script_dir.parent
    datasets = args.datasets or list(configs.keys())

    for name in datasets:
        if name not in configs:
            print(f"[skip] Unknown dataset: {name}")
            continue
        cfg = configs[name]
        emb_dir = proj_root / cfg["embeddings_dir"]
        if not emb_dir.exists() and args.skip_missing:
            print(f"[skip] {name}: embeddings not found at {emb_dir}")
            continue
        print(f"\n{'='*60}")
        print(f"Running: {cfg['name']}")
        print(f"{'='*60}")
        ret = subprocess.run(
            [
                sys.executable,
                str(script_dir / "run_dataset.py"),
                "--dataset", name,
                "--checkpoint", checkpoint,
            ],
            cwd=str(proj_root),
        )
        if ret.returncode != 0:
            print(f"[FAIL] {name} exited with code {ret.returncode}")
        else:
            print(f"[OK] {name}")

    # Consolidated view
    print(f"\n{'='*60}")
    print("Running consolidated results summary...")
    print(f"{'='*60}")
    ret = subprocess.run(
        [sys.executable, str(script_dir / "collect_cross_dataset_results.py")],
        cwd=str(proj_root),
    )
    return ret.returncode


if __name__ == "__main__":
    sys.exit(main())
