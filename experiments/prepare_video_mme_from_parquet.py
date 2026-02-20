#!/usr/bin/env python3
"""
Prepare VideoMME annotations from parquet file.
Reads the parquet mapping, extracts videos from zip if needed, and writes annotations.json.

Expected layout in extdataset/video_mme/:
  - *.parquet                    # Mapping file (video_id, question, options, answer)
  - videos_chunked_01.zip        # Video zip(s) - will extract to videos/
  - subtitle.zip                 # Optional

Parquet columns (HuggingFace lmms-lab/Video-MME):
  - video_id (str, 3 chars) or videoID (str)
  - question (str)
  - options (list of 4 strings)
  - answer (A, B, C, or D)

Usage:
  python experiments/prepare_video_mme_from_parquet.py --data-dir extdataset/video_mme
  python experiments/prepare_video_mme_from_parquet.py --data-dir extdataset/video_mme --extract-zips
"""

import argparse
import json
import zipfile
from pathlib import Path
from typing import Optional


def find_parquet(data_dir: Path) -> Optional[Path]:
    """Find first .parquet file in data dir."""
    for p in data_dir.glob("*.parquet"):
        return p
    for p in data_dir.rglob("*.parquet"):
        return p
    return None


def extract_video_zips(data_dir: Path, videos_dir: Path) -> int:
    """Extract video zip files to videos_dir. Returns number of videos extracted."""
    extracted = 0
    for z in sorted(data_dir.glob("*.zip")):
        if "subtitle" in z.name.lower():
            continue
        print(f"  Extracting {z.name}...")
        with zipfile.ZipFile(z, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith((".mp4", ".avi", ".mkv", ".webm")):
                    zf.extract(name, videos_dir)
                    extracted += 1
    return extracted


def load_video_mme_from_parquet(parquet_path: Path) -> list:
    """Load VideoMME samples from parquet. Returns list of {video_id, video_path, text, target}."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for parquet. Install: pip install pandas pyarrow")

    df = pd.read_parquet(parquet_path)

    # Handle different column names
    vid_col = "video_id" if "video_id" in df.columns else "videoID"
    if vid_col not in df.columns:
        vid_col = df.columns[0]  # fallback

    samples = []
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3, "a": 0, "b": 1, "c": 2, "d": 3}

    for _, row in df.iterrows():
        vid = str(row.get(vid_col, row.get("videoID", "")))
        question = str(row.get("question", ""))
        opts = row.get("options", [])
        if opts is None:
            opts = []
        if hasattr(opts, "__iter__") and not isinstance(opts, str):
            opts = [str(o) if not isinstance(o, dict) else o.get("text", str(o)) for o in opts]
        else:
            opts = []
        answer = str(row.get("answer", "A")).strip().upper()
        target = answer_map.get(answer[0] if answer else "A", 0)

        text = question
        if opts:
            text += " " + " ".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts[:4]))

        samples.append({
            "video_id": vid,
            "video_path": f"videos/{vid}.mp4",
            "text": text,
            "target": target,
        })
    return samples


def find_video_file(vid: str, base: Path) -> Optional[Path]:
    """Find video file by id. Search base and subdirs."""
    exts = [".mp4", ".avi", ".mkv", ".webm"]
    for ext in exts:
        cand = base / f"{vid}{ext}"
        if cand.exists():
            return cand
    for p in base.rglob("*"):
        if p.suffix.lower() in exts and (p.stem == vid or p.name.startswith(vid)):
            return p
    return None


def resolve_video_paths(samples: list, data_dir: Path, videos_dir: Path) -> list:
    """Update video_path to match actual file location."""
    for s in samples:
        vid = s["video_id"]
        found = find_video_file(vid, videos_dir)
        if not found and videos_dir != data_dir:
            found = find_video_file(vid, data_dir)
        if found:
            s["video_path"] = str(found.relative_to(data_dir))
        else:
            s["video_path"] = f"videos/{vid}.mp4"
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="extdataset/video_mme", help="VideoMME data directory (extdataset/video_mme)")
    p.add_argument("--output", default=None, help="Output annotations.json path")
    p.add_argument("--extract-zips", action="store_true", help="Extract video zips to videos/")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}")
        print("Create it and add: parquet file, videos_chunked_*.zip")
        return 1

    parquet_path = find_parquet(data_dir)
    if not parquet_path:
        print(f"No .parquet file found in {data_dir}")
        return 1

    print(f"Reading parquet: {parquet_path}")
    samples = load_video_mme_from_parquet(parquet_path)
    print(f"  Loaded {len(samples)} samples")

    videos_dir = data_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    if args.extract_zips:
        n = extract_video_zips(data_dir, videos_dir)
        if n > 0:
            print(f"  Extracted {n} videos to {videos_dir}")

    # Check if videos exist; if not, maybe they're in a subdir or different structure
    if not list(videos_dir.glob("*.*")):
        alt = data_dir / "videos"
        if alt.exists():
            for d in alt.iterdir():
                if d.is_dir() and list(d.glob("*.mp4")):
                    videos_dir = d
                    break

    samples = resolve_video_paths(samples, data_dir, videos_dir)

    out_path = Path(args.output) if args.output else data_dir / "annotations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write format expected by precompute: video_path, text, target
    ann = [{"video_path": s["video_path"], "text": s["text"], "target": s["target"]} for s in samples]
    with open(out_path, "w") as f:
        json.dump(ann, f, indent=2)

    print(f"Wrote {out_path} ({len(ann)} samples)")
    return 0


if __name__ == "__main__":
    exit(main())
