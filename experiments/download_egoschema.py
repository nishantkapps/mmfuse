#!/usr/bin/env python3
"""
Download EgoSchema (ego-centric video QA) from HuggingFace and prepare for MMFuse.

EgoSchema is the ego-to-exo benchmark used by VisCoP for egocentric video understanding.
Uses the 500-question Subset with public answers for offline evaluation.

Output: extdataset/egoschema/annotations.json, videos/*.mp4

After download, run precompute and evaluation:
  python experiments/precompute_egoschema.py --out-dir embeddings/egoschema
  python experiments/run_dataset.py --dataset egoschema --checkpoint checkpoints/ckpt_sdata_epoch_N.pt --linear-probe

Usage:
  python experiments/download_egoschema.py --max-samples 500
  python experiments/download_egoschema.py --video-zips 2
  python experiments/download_egoschema.py --skip-videos  # annotations only

Requires: pip install pandas pyarrow huggingface_hub
"""

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent


def _strip_option_prefix(s: str) -> str:
    """Strip leading 'A. ', 'B. ', etc from option text."""
    if not s or not isinstance(s, str):
        return str(s) if s else ""
    m = re.match(r"^[A-Ea-e]\.\s*", s.strip())
    return s[m.end() :].strip() if m else s.strip()


def download_egoschema(
    out_dir: Path,
    max_samples: int = 500,
    video_zips: int = 2,
    skip_videos: bool = False,
) -> bool:
    """Download EgoSchema Subset (500 QA with answers) + videos from HuggingFace."""
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install pandas pyarrow huggingface_hub")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    ann_path = out_dir / "annotations.json"
    if ann_path.exists():
        with open(ann_path) as f:
            existing = json.load(f)
        if len(existing) >= max_samples:
            print(f"EgoSchema: annotations.json exists ({len(existing)} samples), skipping")
            n_vids = len(list(videos_dir.glob("*.mp4")))
            print(f"  Videos: {n_vids}")
            return True

    # Subset parquet has 500 questions with answers (0-4)
    print("EgoSchema: loading Subset annotations from HuggingFace...")
    try:
        parquet_path = hf_hub_download(
            repo_id="lmms-lab/egoschema",
            filename="Subset/test-00000-of-00001.parquet",
            repo_type="dataset",
        )
    except Exception as e:
        print(f"  Failed to download Subset parquet: {e}")
        return False

    df = pd.read_parquet(parquet_path)
    n = min(max_samples, len(df))

    rows = []
    for _, row in df.head(n).iterrows():
        vid = str(row.get("video_idx", row.get("video_id", "")))
        if not vid:
            continue
        question = str(row.get("question", ""))
        opts_raw = row.get("option", [])
        if opts_raw is None:
            opts_raw = []
        opts = [str(o) for o in opts_raw] if hasattr(opts_raw, "__iter__") and not isinstance(opts_raw, str) else []
        opts_clean = [_strip_option_prefix(o) for o in opts[:5]]
        ans_raw = row.get("answer")
        if ans_raw is None or (isinstance(ans_raw, float) and str(ans_raw) == "nan"):
            continue
        target = int(ans_raw) if isinstance(ans_raw, (int, float)) else int(str(ans_raw).strip())
        target = max(0, min(4, target))
        answer = chr(65 + target)

        rows.append({
            "video_id": vid,
            "video_path": f"videos/{vid}.mp4",
            "question": question,
            "options": opts_clean,
            "answer": answer,
            "target": target,
        })

    if not rows:
        print("No samples with answers found.")
        return False

    with open(ann_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved {len(rows)} samples to annotations.json")

    # Video zips (videos_chunked_01.zip .. 05.zip)
    if not skip_videos and video_zips > 0:
        for i in range(1, min(video_zips + 1, 6)):
            zip_name = f"videos_chunked_{i:02d}.zip"
            zip_path = out_dir / zip_name
            if zip_path.exists():
                print(f"  {zip_name} exists, skipping download")
            else:
                print(f"  Downloading {zip_name} (~5GB each)...")
                try:
                    hf_hub_download(
                        repo_id="lmms-lab/egoschema",
                        filename=zip_name,
                        repo_type="dataset",
                        local_dir=out_dir,
                        local_dir_use_symlinks=False,
                    )
                except Exception as e:
                    print(f"  Failed: {e}")
                    continue

            if zip_path.exists():
                n_vids_before = len(list(videos_dir.glob("*.mp4")))
                if n_vids_before >= len(rows):
                    print(f"  All {len(rows)} videos present, skipping extract")
                else:
                    print(f"  Extracting {zip_name}...")
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        for name in zf.namelist():
                            if name.lower().endswith(".mp4"):
                                zf.extract(name, out_dir)
                    # Flatten: any *.mp4 in out_dir tree -> videos/*.mp4
                    for f in out_dir.rglob("*.mp4"):
                        if f.parent != videos_dir:
                            dest = videos_dir / f.name
                            if not dest.exists():
                                shutil.move(str(f), str(dest))
                            try:
                                f.unlink(missing_ok=True)
                            except Exception:
                                pass
                    for d in sorted(out_dir.rglob("*"), key=lambda x: -len(x.parts)):
                        if d.is_dir() and d != videos_dir and d != out_dir and not any(d.iterdir()):
                            try:
                                d.rmdir()
                            except Exception:
                                pass
                    print(f"  Extracted to {videos_dir}")

    n_vids = len(list(videos_dir.glob("*.mp4")))
    print(f"EgoSchema done. Saved {len(rows)} samples. Videos: {n_vids}")
    return True


def main():
    p = argparse.ArgumentParser(description="Download EgoSchema (ego-centric video QA) for VisCoP evaluation")
    p.add_argument("--out-dir", default=None, help="Output dir (default: extdataset/egoschema)")
    p.add_argument("--max-samples", type=int, default=500, help="Max samples (Subset has 500)")
    p.add_argument("--video-zips", type=int, default=2,
                   help="Number of video zips to download (1-5, ~5GB each)")
    p.add_argument("--skip-videos", action="store_true",
                   help="Skip video download (annotations only)")

    args = p.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else _proj_root / "extdataset" / "egoschema"

    ok = download_egoschema(
        out_dir,
        max_samples=args.max_samples,
        video_zips=args.video_zips,
        skip_videos=args.skip_videos,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
