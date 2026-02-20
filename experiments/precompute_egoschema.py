#!/usr/bin/env python3
"""
Precompute embeddings for EgoSchema.
Expected structure in extdataset/egoschema/:
  - videos/ with video files
  - annotations.json with: video_id, question, options (A-E), answer

Usage:
  python experiments/precompute_egoschema.py --out-dir embeddings/egoschema
"""

import argparse
import json
import sys
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

from experiments.precompute_video_text import precompute_from_samples  # noqa: E402

ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "a": 0, "b": 1, "c": 2, "d": 3, "e": 4}


def load_egoschema_samples(data_dir: Path):
    """Load EgoSchema samples."""
    ann_path = data_dir / "annotations.json"
    if not ann_path.exists():
        return []

    with open(ann_path) as f:
        data = json.load(f)

    samples = []
    video_dir = data_dir / "videos"
    for item in (data if isinstance(data, list) else data.get("questions", [data])):
        vid = item.get("video_id", item.get("video", ""))
        question = item.get("question", "")
        options = item.get("options", [])
        answer = item.get("answer", "A")
        target = ANSWER_MAP.get(str(answer).strip().upper(), 0)
        text = question + " " + " ".join(f"({chr(65+i)}) {o}" for i, o in enumerate(options[:5]) if o)
        rel_path = f"videos/{vid}.mp4" if not Path(vid).suffix else str(vid)
        samples.append({"video_path": rel_path, "text": text, "target": target})
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--vision-encoder", choices=["viscop", "clip"], default="viscop")
    p.add_argument("--text-encoder", choices=["clip", "bert"], default="clip")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _proj_root / "extdataset" / "egoschema"
    out_dir = Path(args.out_dir)

    if (out_dir / "config.json").exists() and any(out_dir.glob("*.pt")):
        print(f"Embeddings already exist in {out_dir}. Skipping precompute.")
        return 0

    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}. Create extdataset/egoschema/ and add data.")
        return 1

    samples = load_egoschema_samples(data_dir)
    if not samples:
        print("No samples found. Add annotations.json with video_id, question, options, answer.")
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
