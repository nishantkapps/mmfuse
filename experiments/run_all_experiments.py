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


def find_checkpoint():
    """Find latest checkpoint from model training (excludes ablation_runs)."""
    proj = Path(__file__).resolve().parent.parent
    # 1. checkpoints/ from training/train_sdata_attention.py
    candidates = sorted(proj.glob("checkpoints/ckpt_sdata_epoch_*.pt"))
    if candidates:
        return str(candidates[-1])
    # 2. runs/ subdirs
    candidates = sorted(proj.glob("runs/*/ckpt_sdata_epoch_*.pt"))
    if candidates:
        return str(candidates[-1])
    # 3. Exported model
    if (proj / "models/sdata_viscop/pytorch_model.bin").exists():
        return str(proj / "models/sdata_viscop/pytorch_model.bin")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pt); auto-picks latest from training if omitted")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Specific datasets to run (default: all)")
    p.add_argument("--skip-missing", action="store_true", default=True,
                   help="Skip datasets without embeddings (default: True)")
    args = p.parse_args()

    checkpoint = args.checkpoint or find_checkpoint()
    if not checkpoint or not Path(checkpoint).exists():
        print("No model training checkpoint found.")
        print("  Train first: python training/train_sdata_attention.py --use-precomputed --embeddings-dir embeddings/sdata_viscop")
        print("  Or pass: python experiments/run_all_experiments.py --checkpoint path/to/ckpt_sdata_epoch_N.pt")
        return 1

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
