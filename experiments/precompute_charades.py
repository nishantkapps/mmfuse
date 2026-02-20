#!/usr/bin/env python3
"""
Precompute embeddings for Charades (ADL-X).
Expected structure in extdataset/charades/:
  - videos/ or Charades_v1_480/ with *.mp4 files named <video_id>.mp4
  - annotations.json or Charades_v1_train.csv + Charades_v1_test.csv
  - annotations format: [{video_path, text (description), target (0-156)}]
  - Or CSV: id,actions (e.g. "c001 c005" -> take first as primary class)

Usage:
  python experiments/precompute_charades.py --out-dir embeddings/charades
"""

import argparse
import csv
import json
import sys
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

from experiments.precompute_video_text import precompute_from_samples, load_frame  # noqa: E402

# Charades action index: c001 -> 0, c002 -> 1, ... c157 -> 156
CHARADES_ACTIONS = [f"c{i:03d}" for i in range(1, 158)]


def load_charades_samples(data_dir: Path):
    """Load Charades samples. Supports annotations.json or Charades CSV format."""
    ann_path = data_dir / "annotations.json"
    if ann_path.exists():
        with open(ann_path) as f:
            return json.load(f)

    # Charades native format: Charades_v1_train.csv, Charades_v1_test.csv
    samples = []
    video_dir = data_dir / "videos"
    if not video_dir.exists():
        video_dir = data_dir / "Charades_v1_480"
    for csv_name in ["Charades_v1_train.csv", "Charades_v1_test.csv"]:
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row.get("id", row.get("video_id", ""))
                actions = row.get("actions", row.get("action", ""))
                # actions: "c001 c005 c012" - take first as primary
                action_ids = actions.split() if isinstance(actions, str) else []
                try:
                    label = CHARADES_ACTIONS.index(action_ids[0]) if action_ids else 0
                except ValueError:
                    label = 0
                rel = f"{video_dir.name}/{vid}.mp4"
                samples.append({
                    "video_path": rel,
                    "text": row.get("description", row.get("caption", f"action {label}")),
                    "target": label,
                })
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None, help="extdataset/charades")
    p.add_argument("--out-dir", required=True, help="embeddings/charades")
    p.add_argument("--vision-encoder", choices=["viscop", "clip"], default="viscop")
    p.add_argument("--text-encoder", choices=["viscop", "clip", "bert"], default="viscop",
                   help="viscop: use VisCoP text encoder (same as vision, recommended)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _proj_root / "extdataset" / "charades"
    out_dir = Path(args.out_dir)

    if (out_dir / "config.json").exists() and any(out_dir.glob("*.pt")):
        print(f"Embeddings already exist in {out_dir}. Skipping precompute.")
        return 0

    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}. Create extdataset/charades/ and add data.")
        return 1

    samples = load_charades_samples(data_dir)
    if not samples:
        print("No samples found. Add annotations.json or Charades_v1_*.csv")
        return 1

    precompute_from_samples(
        samples,
        data_dir,
        out_dir,
        vision_encoder=args.vision_encoder,
        text_encoder=args.text_encoder,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
