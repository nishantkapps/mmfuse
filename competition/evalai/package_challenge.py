#!/usr/bin/env python3
"""
MIRA Challenge — Package EvalAI Challenge Zip

Takes the ground truth CSVs produced by prepare_dataset.py and packages
everything into a single zip file that can be uploaded to EvalAI via:
  EvalAI Dashboard → Host Challenge → Upload zip

Usage:
  python competition/evalai/package_challenge.py \
      --data-dir competition/data \
      --output mira_challenge.zip

Prerequisites:
  1. Run competition/prepare_dataset.py to generate ground truth CSVs
  2. Run competition/extract_trajectories.py to fill dx/dy columns
  3. Place your challenge logo at competition/evalai/logo/challenge_logo.png
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path


EVALAI_DIR = Path(__file__).resolve().parent


def split_ground_truth_for_dev(gt_path, dev_fraction=0.3, seed=42):
    """
    Take the full test ground truth and split it into a dev subset
    (for the dev phase leaderboard) and full test (for the test phase).
    Returns (dev_lines, test_lines) including the header.
    """
    import random
    rng = random.Random(seed)

    with open(gt_path) as f:
        lines = f.readlines()
    header = lines[0]
    data = lines[1:]
    rng.shuffle(data)
    n_dev = max(1, int(len(data) * dev_fraction))
    return header, data[:n_dev], data


def main():
    parser = argparse.ArgumentParser(description="Package MIRA challenge for EvalAI")
    parser.add_argument("--data-dir", default="competition/data",
                        help="Directory with ground truth CSVs from prepare_dataset.py")
    parser.add_argument("--output", default="mira_challenge.zip",
                        help="Output zip file path")
    parser.add_argument("--dev-fraction", type=float, default=0.3,
                        help="Fraction of test set to use as dev leaderboard")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    gt_file = data_dir / "test_ground_truth.csv"

    if not gt_file.exists():
        print(f"ERROR: {gt_file} not found. Run prepare_dataset.py and extract_trajectories.py first.")
        return

    annotations_dir = EVALAI_DIR / "annotations"
    annotations_dir.mkdir(exist_ok=True)

    header, dev_lines, full_lines = split_ground_truth_for_dev(
        gt_file, dev_fraction=args.dev_fraction
    )

    for prefix in ["track1", "track2", "combined"]:
        (annotations_dir / f"{prefix}_dev_gt.csv").write_text(header + "".join(dev_lines))
        (annotations_dir / f"{prefix}_test_gt.csv").write_text(header + "".join(full_lines))

    print(f"Annotations: {len(dev_lines)} dev samples, {len(full_lines)} test samples")

    logo_path = EVALAI_DIR / "logo" / "challenge_logo.png"
    if not logo_path.exists():
        print(f"WARNING: {logo_path} not found — creating a placeholder.")
        logo_path.write_bytes(b"")

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(EVALAI_DIR):
            root_path = Path(root)
            if "__pycache__" in str(root_path) or root_path.name == "__pycache__":
                continue
            if root_path.name == "package_challenge.py":
                continue
            for fname in files:
                if fname.endswith(".pyc") or fname == "package_challenge.py":
                    continue
                full = root_path / fname
                arcname = full.relative_to(EVALAI_DIR)
                zf.write(full, arcname)

    print(f"\nChallenge zip created: {args.output}")
    print(f"Upload this zip at: https://eval.ai/web/challenge-host/create")

    print("\nContents:")
    with zipfile.ZipFile(args.output, "r") as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            print(f"  {info.filename:50s} {info.file_size:>8d} bytes")


if __name__ == "__main__":
    main()
