#!/usr/bin/env python3
"""
Precompute embeddings for NeXTQA.
Expected structure in extdataset/nextqa/:
  - videos/ with video files
  - annotations.json or nextqa_train.json / nextqa_val.json with:
    video_id, question, a0..a4 (options), answer (0-4)

Usage:
  python experiments/precompute_nextqa.py --out-dir embeddings/nextqa
"""

import argparse
import json
import sys
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

from experiments.precompute_video_text import precompute_from_samples  # noqa: E402


def load_nextqa_samples(data_dir: Path):
    """Load NeXTQA samples."""
    samples = []
    video_dir = data_dir / "videos"
    for ann_name in ["annotations.json", "nextqa_train.json", "nextqa_val.json", "train.json", "val.json"]:
        ann_path = data_dir / ann_name
        if not ann_path.exists():
            continue
        with open(ann_path) as f:
            data = json.load(f)
        for item in (data if isinstance(data, list) else data.get("questions", [data])):
            vid = item.get("video_id", item.get("video", ""))
            question = item.get("question", "")
            opts = [item.get(f"a{i}", item.get(f"option_{i}", "")) for i in range(5)]
            answer = int(item.get("answer", item.get("correct", 0)))
            text = question + " " + " ".join(f"({i}) {o}" for i, o in enumerate(opts) if o)
            rel_path = f"videos/{vid}.mp4" if not Path(vid).suffix else str(vid)
            samples.append({"video_path": rel_path, "text": text, "target": answer})
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--vision-encoder", choices=["viscop", "clip"], default="viscop")
    p.add_argument("--text-encoder", choices=["viscop", "clip", "bert"], default="viscop",
                   help="viscop: use VisCoP text encoder (same as vision, recommended)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _proj_root / "extdataset" / "nextqa"
    out_dir = Path(args.out_dir)

    if (out_dir / "config.json").exists() and any(out_dir.glob("*.pt")):
        print(f"Embeddings already exist in {out_dir}. Skipping precompute.")
        return 0

    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}. Create extdataset/nextqa/ and add data.")
        return 1

    samples = load_nextqa_samples(data_dir)
    if not samples:
        print("No samples found. Add annotations.json or nextqa_*.json")
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
