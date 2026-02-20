#!/usr/bin/env python3
"""
Precompute embeddings for VideoMME.
Expected structure in extdataset/video_mme/:
  - videos/ with video files
  - annotations.json with format:
    [{"video_id": "...", "question": "...", "options": ["A", "B", "C", "D"], "answer": "A"}, ...]
  - answer maps to target: A->0, B->1, C->2, D->3

Usage:
  python experiments/precompute_video_mme.py --out-dir embeddings/video_mme
"""

import argparse
import json
import sys
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

from experiments.precompute_video_text import precompute_from_samples  # noqa: E402

ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "a": 0, "b": 1, "c": 2, "d": 3}


def load_video_mme_samples(data_dir: Path):
    """Load VideoMME samples from annotations.json or native format."""
    ann_path = data_dir / "annotations.json"
    if not ann_path.exists():
        return []

    with open(ann_path) as f:
        data = json.load(f)

    samples = []
    video_dir = data_dir / "videos"
    for item in data if isinstance(data, list) else data.get("questions", [data]):
        if "video_path" in item and "text" in item and "target" in item:
            samples.append({"video_path": item["video_path"], "text": item["text"], "target": int(item["target"])})
            continue
        vid = item.get("video_id", item.get("video", ""))
        question = item.get("question", item.get("question_text", ""))
        options = item.get("options", [])
        answer = item.get("answer", item.get("correct_answer", "A"))
        target = ANSWER_MAP.get(str(answer).strip().upper(), 0)
        # Build text: question + options for context
        text = question
        if options:
            text += " " + " ".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(options[:4]))
        video_path = video_dir / vid
        if not video_path.suffix:
            for ext in [".mp4", ".avi", ".mkv"]:
                if (video_dir / f"{vid}{ext}").exists():
                    video_path = video_dir / f"{vid}{ext}"
                    break
        rel_path = str(video_path.relative_to(data_dir)) if video_path.exists() else f"videos/{vid}.mp4"
        samples.append({"video_path": rel_path, "text": text, "target": target})
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--vision-encoder", choices=["viscop", "clip"], default="viscop")
    p.add_argument("--text-encoder", choices=["viscop", "clip", "bert"], default="viscop",
                   help="viscop: use VisCoP text encoder (same model as vision, recommended)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _proj_root / "extdataset" / "video_mme"
    out_dir = Path(args.out_dir)

    if (out_dir / "config.json").exists() and any(out_dir.glob("*.pt")):
        print(f"Embeddings already exist in {out_dir}. Skipping precompute.")
        return 0

    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}. Create extdataset/video_mme/ and add data.")
        return 1

    samples = load_video_mme_samples(data_dir)
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
