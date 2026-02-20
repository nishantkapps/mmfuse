#!/usr/bin/env python3
"""
Download annotations + videos for VideoMME, NeXTQA, Charades, VIMA-Bench.
Uses smaller defaults (~2000 samples) for reasonable disk use.

Usage:
  python experiments/download_datasets_full.py all
  python experiments/download_datasets_full.py video_mme --max-samples 2000 --video-zips 5
  python experiments/download_datasets_full.py nextqa --max-samples 3000
  python experiments/download_datasets_full.py charades --max-samples 1000
  python experiments/download_datasets_full.py vima_bench --max-samples 500
  python experiments/download_datasets_full.py egoschema --max-samples 500 --video-zips 2

VIMA-Bench: downloads ~21.5 GB zip from HuggingFace. For L1/L2/L3 evaluation.
EgoSchema: ego-centric video QA (VisCoP benchmark). Subset 500 with answers from HuggingFace.

Requires: pip install pandas pyarrow huggingface_hub
For NeXTQA videos: pip install gdown (optional)
"""

import argparse
import ast
import json
import shutil
import zipfile
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent

# Defaults: enough for good linear probe, not full 50k
DEFAULTS = {
    "video_mme": {"max_samples": 2000, "video_zips": 5},  # ~25GB, ~150 videos
    "nextqa": {"max_samples": 3000, "video_zips": 0},
    "charades": {"max_samples": 1000, "video_zips": 0},
    "vima_bench": {"max_samples": 500},  # L1/L2/L3 from VIMA-Data (21.5 GB zip)
    "egoschema": {"max_samples": 500, "video_zips": 2},  # Ego-centric QA (VisCoP), Subset 500
}


def download_video_mme(out_dir: Path, max_samples: int, video_zips: int) -> bool:
    """Download VideoMME: annotations + video zips from HuggingFace."""
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
            print(f"VideoMME: annotations.json exists ({len(existing)} samples), skipping")
            n_vids = len(list(videos_dir.glob("*.mp4"))) + len(list(videos_dir.rglob("*.mp4")))
            print(f"  Videos: {n_vids}")
            return True

    # Annotations: load parquet directly (avoids datasets LocalFileSystem bug)
    print("VideoMME: loading annotations...")
    parquet_path = None
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files("lmms-lab/Video-MME", repo_type="dataset", revision="refs/convert/parquet")
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if parquet_files:
            parquet_path = hf_hub_download(
                repo_id="lmms-lab/Video-MME",
                filename=parquet_files[0],
                repo_type="dataset",
                revision="refs/convert/parquet",
            )
    except Exception:
        pass
    if parquet_path:
        df = pd.read_parquet(parquet_path)
    else:
        # Fallback: download and read
        import io
        import urllib.request
        url = "https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/refs%2Fconvert%2Fparquet/videomme/test-00000-of-00001.parquet"
        req = urllib.request.Request(url, headers={"User-Agent": "mmfuse-download"})
        with urllib.request.urlopen(req) as r:
            df = pd.read_parquet(io.BytesIO(r.read()))
    n = min(max_samples, len(df))
    df = df.head(n)

    vid_col = "video_id" if "video_id" in df.columns else "videoID"
    ans_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    rows = []
    for _, row in df.iterrows():
        vid = str(row.get(vid_col, row.get("videoID", "")))
        q = str(row.get("question", ""))
        opts = row.get("options", [])
        if opts is not None and hasattr(opts, "__iter__") and not isinstance(opts, str):
            opts = [str(o) if not isinstance(o, dict) else o.get("text", str(o)) for o in opts][:4]
        else:
            opts = []
        ans = str(row.get("answer", "A")).strip().upper()
        target = ans_map.get(ans[0] if ans else "A", 0)
        rows.append({
            "video_id": vid,
            "video_path": f"videos/{vid}.mp4",
            "question": q,
            "options": opts,
            "answer": ans,
            "target": target,
        })
    with open(out_dir / "annotations.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved {len(rows)} samples to annotations.json")

    # Video zips
    if video_zips > 0:
        for i in range(1, min(video_zips + 1, 21)):
            zip_name = f"videos_chunked_{i:02d}.zip"
            zip_path = out_dir / zip_name
            if zip_path.exists():
                print(f"  {zip_name} exists, skipping")
            else:
                print(f"  Downloading {zip_name} (~5GB)...")
                try:
                    hf_hub_download(
                        repo_id="lmms-lab/Video-MME",
                        filename=zip_name,
                        repo_type="dataset",
                        local_dir=out_dir,
                        local_dir_use_symlinks=False,
                    )
                except Exception as e:
                    print(f"  Failed: {e}")
                    continue

            if zip_path.exists():
                n_vids_before = len(list(videos_dir.glob("*.mp4"))) + len(list(videos_dir.rglob("*.mp4")))
                if n_vids_before > 0:
                    print(f"  {zip_name} exists, videos present ({n_vids_before}), skipping extract")
                else:
                    print(f"  Extracting {zip_name}...")
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        for name in zf.namelist():
                            if name.lower().endswith((".mp4", ".avi", ".mkv", ".webm")):
                                zf.extract(name, out_dir)
                    # Flatten: any *.mp4 in out_dir tree -> videos/*.mp4
                    for f in out_dir.rglob("*.mp4"):
                        if f.parent != videos_dir:
                            dest = videos_dir / f.name
                            if not dest.exists():
                                shutil.move(str(f), str(dest))
                            f.unlink(missing_ok=True)
                    for d in sorted(out_dir.rglob("*"), key=lambda x: -len(x.parts)):
                        if d.is_dir() and d != videos_dir and not any(d.iterdir()):
                            d.rmdir()
                    print(f"  Extracted to {videos_dir}")

    n_vids = len(list(videos_dir.glob("*.mp4"))) + len(list(videos_dir.rglob("*.mp4")))
    print(f"VideoMME done. Videos in {videos_dir}: {n_vids}")
    return True


def download_nextqa(out_dir: Path, max_samples: int) -> bool:
    """Download NeXTQA: annotations from HF (parquet). Videos from Google Drive (optional)."""
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download, list_repo_files
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
            print(f"NeXTQA: annotations.json exists ({len(existing)} samples), skipping")
            n_vids = len(list(videos_dir.glob("*.mp4"))) + len(list(videos_dir.rglob("*.mp4")))
            print(f"  Videos: {n_vids}")
            return True

    print("NeXTQA: loading annotations from parquet...")
    parquet_path = None
    for candidate in [
        "MC/test-00000-of-00001.parquet",
        "MC/train-00000-of-00001.parquet",
        "MC/val-00000-of-00001.parquet",
    ]:
        try:
            parquet_path = hf_hub_download(
                repo_id="lmms-lab/NExTQA",
                filename=candidate,
                repo_type="dataset",
                revision="refs/convert/parquet",
            )
            break
        except Exception:
            continue
    if not parquet_path:
        try:
            files = list_repo_files("lmms-lab/NExTQA", repo_type="dataset", revision="refs/convert/parquet")
            parquet_files = [f for f in files if f.endswith(".parquet")]
            if parquet_files:
                parquet_path = hf_hub_download(
                    repo_id="lmms-lab/NExTQA",
                    filename=parquet_files[0],
                    repo_type="dataset",
                    revision="refs/convert/parquet",
                )
        except Exception:
            pass
    if parquet_path:
        df = pd.read_parquet(parquet_path)
    else:
        import io
        import urllib.request
        url = "https://huggingface.co/datasets/lmms-lab/NExTQA/resolve/refs%2Fconvert%2Fparquet/MC/test-00000-of-00001.parquet"
        req = urllib.request.Request(url, headers={"User-Agent": "mmfuse-download"})
        with urllib.request.urlopen(req) as r:
            df = pd.read_parquet(io.BytesIO(r.read()))
    n = min(max_samples, len(df))
    df = df.head(n)

    vid_col = "video_id" if "video_id" in df.columns else "video"
    rows = []
    for _, row in df.iterrows():
        vid = str(row.get(vid_col, row.get("video", "")))
        q = str(row.get("question", ""))
        a0 = str(row.get("a0", row.get("option_0", "")))
        a1 = str(row.get("a1", row.get("option_1", "")))
        a2 = str(row.get("a2", row.get("option_2", "")))
        a3 = str(row.get("a3", row.get("option_3", "")))
        a4 = str(row.get("a4", row.get("option_4", "")))
        ans = int(row.get("answer", row.get("correct", 0)))
        rows.append({
            "video_id": vid,
            "video_path": f"videos/{vid}.mp4",
            "question": q,
            "a0": a0, "a1": a1, "a2": a2, "a3": a3, "a4": a4,
            "answer": ans,
        })
    with open(ann_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved {len(rows)} samples to annotations.json")

    # Videos: Google Drive (NExTVideo). Requires gdown. Skip if videos exist.
    n_vids = len(list(videos_dir.glob("*.mp4"))) + len(list(videos_dir.rglob("*.mp4")))
    if n_vids > 0:
        print(f"  NeXTQA videos: {n_vids} already present, skipping download")
    else:
        try:
            import gdown
            drive_id = "1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH"
            zip_path = out_dir / "nextqa_videos.zip"
            if not zip_path.exists():
                print("  NeXTQA videos: downloading from Google Drive (~50GB)...")
                gdown.download(id=drive_id, output=str(zip_path), quiet=False, fuzzy=True)
            if zip_path.exists():
                print("  Extracting videos...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(videos_dir)
                print("  Done.")
        except ImportError:
            print("  NeXTQA videos: pip install gdown, then re-run. Or download manually:")
            print("    https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH")
            print("    Extract to extdataset/nextqa/videos/")
        except Exception as e:
            print(f"  Video download failed: {e}")

    n_vids = len(list(videos_dir.glob("*.mp4"))) + len(list(videos_dir.rglob("*.mp4")))
    print(f"NeXTQA done. Videos: {n_vids}")
    return True


def download_charades(out_dir: Path, max_samples: int) -> bool:
    """Download Charades: annotations from parquet. Videos from HuggingFace cache or manual."""
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Install: pip install pandas pyarrow huggingface_hub")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    ann_path = out_dir / "annotations.json"
    skip_annotations = False
    if ann_path.exists():
        with open(ann_path) as f:
            existing = json.load(f)
        if len(existing) >= max_samples:
            print(f"Charades: annotations.json exists ({len(existing)} samples), skipping annotations")
            skip_annotations = True
            rows = existing
        else:
            skip_annotations = False

    if not skip_annotations:
        print("Charades: loading annotations from parquet (train+test+validation)...")
    dfs = []
    for candidate in [
        "default/train/train-00000-of-00001.parquet",
        "default/test/test-00000-of-00001.parquet",
        "default/validation/validation-00000-of-00001.parquet",
    ]:
        try:
            parquet_path = hf_hub_download(
                repo_id="Aditya02/Charades-Action-Sequence-Sample",
                filename=candidate,
                repo_type="dataset",
                revision="refs/convert/parquet",
            )
            dfs.append(pd.read_parquet(parquet_path))
        except Exception:
            continue
    if not dfs:
        try:
            files = list_repo_files("Aditya02/Charades-Action-Sequence-Sample", repo_type="dataset", revision="refs/convert/parquet")
            parquet_files = [f for f in files if f.endswith(".parquet")]
            for pf in parquet_files[:3]:
                try:
                    p = hf_hub_download(
                        repo_id="Aditya02/Charades-Action-Sequence-Sample",
                        filename=pf,
                        repo_type="dataset",
                        revision="refs/convert/parquet",
                    )
                    dfs.append(pd.read_parquet(p))
                except Exception:
                    continue
        except Exception:
            pass
    if not skip_annotations and not dfs:
        print("  Failed to load Charades parquet")
        return False

    if not skip_annotations:
        df = pd.concat(dfs, ignore_index=True)
        n = min(max_samples, len(df))
        df = df.head(n)

        rows = []
        for _, row in df.iterrows():
            vid = str(row.get("video_id", row.get("id", "")))
            labels_raw = row.get("labels", [])
            if labels_raw is None:
                labels = []
            elif isinstance(labels_raw, str):
                try:
                    labels = ast.literal_eval(labels_raw) if labels_raw else []
                except Exception:
                    labels = []
            else:
                try:
                    labels = list(labels_raw)
                except (TypeError, ValueError):
                    labels = []
            if not isinstance(labels, list):
                labels = []
            target = int(labels[0]) if labels else 0
            text = str(row.get("script", row.get("descriptions", "")))
            if isinstance(text, (list, tuple)):
                text = text[0] if text else ""
            rows.append({
                "video_id": vid,
                "video_path": f"videos/{vid}.mp4",
                "text": text or "",
                "target": target,
            })

        with open(ann_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"  Saved {len(rows)} samples to annotations.json")

    n_vids = len(list(videos_dir.glob("*.mp4")))
    if n_vids == 0:
        # Try HuggingFace (faster than Allen AI) - 4 parts, ~16 GB total
        try:
            from huggingface_hub import hf_hub_download
            for i in range(1, 5):
                zip_name = f"Charades_v1_480_part_{i}.zip"
                zip_path = out_dir / zip_name
                n_before = len(list(videos_dir.glob("*.mp4")))
                if n_before > 0 and i > 1:
                    print(f"  Videos present ({n_before}), skipping remaining zips")
                    break
                if zip_path.exists():
                    print(f"  {zip_name} exists, extracting...")
                else:
                    print(f"  Downloading {zip_name} (~5GB, HuggingFace CDN)...")
                    hf_hub_download(
                        repo_id="lmms-lab/charades_sta",
                        filename=zip_name,
                        repo_type="dataset",
                        local_dir=out_dir,
                        local_dir_use_symlinks=False,
                    )
                if zip_path.exists():
                    print(f"  Extracting {zip_name}...")
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        for name in zf.namelist():
                            if name.lower().endswith(".mp4"):
                                zf.extract(name, out_dir)
                    # Flatten to videos/
                    for f in out_dir.rglob("*.mp4"):
                        if f.parent != videos_dir:
                            dest = videos_dir / f.name
                            if not dest.exists():
                                shutil.move(str(f), str(dest))
                            f.unlink(missing_ok=True)
                    for d in sorted(out_dir.rglob("*"), key=lambda x: -len(x.parts)):
                        if d.is_dir() and d != videos_dir and not any(d.iterdir()):
                            d.rmdir()
        except Exception as e:
            print(f"  HuggingFace download failed: {e}")
            print("  Alternatives:")
            print("    1. Direct S3 (often faster): wget https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip")
            print("    2. Allen AI: https://prior.allenai.org/projects/charades")
            print(f"  Extract to {out_dir}/, then: cp Charades_v1_480/*.mp4 {videos_dir}/")

    n_vids = len(list(videos_dir.glob("*.mp4")))
    print(f"Charades done. Saved {len(rows)} samples. Videos: {n_vids}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["video_mme", "nextqa", "charades", "vima_bench", "egoschema", "all"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--video-zips", type=int, default=None,
                   help="VideoMME: 1-21 (default 5). EgoSchema: 1-5 (default 2). 0 = skip videos.")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    ext = _proj_root / "extdataset"
    ext.mkdir(exist_ok=True)

    names = ["video_mme", "nextqa", "charades", "vima_bench", "egoschema"] if args.dataset == "all" else [args.dataset]
    ok = 0

    for name in names:
        cfg = DEFAULTS[name]
        max_samples = args.max_samples if args.max_samples is not None else cfg["max_samples"]
        video_zips = args.video_zips if args.video_zips is not None else cfg.get("video_zips", 0)
        out = Path(args.out_dir) if args.out_dir else ext / name

        print(f"\n{'='*60}\n{name.upper()}\n{'='*60}")
        if name == "video_mme":
            if download_video_mme(out, max_samples, video_zips):
                ok += 1
        elif name == "nextqa":
            if download_nextqa(out, max_samples):
                ok += 1
        elif name == "charades":
            if download_charades(out, max_samples):
                ok += 1
        elif name == "vima_bench":
            import subprocess
            import sys
            cfg = DEFAULTS.get("vima_bench", {"max_samples": 500})
            max_vima = args.max_samples if args.max_samples is not None else cfg["max_samples"]
            r = subprocess.run(
                [sys.executable, str(_proj_root / "experiments" / "download_vima_bench.py"),
                 "--out-dir", str(out), "--max-samples", str(max_vima)],
                cwd=str(_proj_root),
            )
            if r.returncode == 0:
                ok += 1
        elif name == "egoschema":
            import subprocess
            import sys
            cfg = DEFAULTS.get("egoschema", {"max_samples": 500, "video_zips": 2})
            max_ego = args.max_samples if args.max_samples is not None else cfg["max_samples"]
            video_zips = args.video_zips if args.video_zips is not None else cfg.get("video_zips", 2)
            cmd = [sys.executable, str(_proj_root / "experiments" / "download_egoschema.py"),
                   "--out-dir", str(out), "--max-samples", str(max_ego)]
            if video_zips > 0:
                cmd.extend(["--video-zips", str(video_zips)])
            else:
                cmd.append("--skip-videos")
            r = subprocess.run(cmd, cwd=str(_proj_root))
            if r.returncode == 0:
                ok += 1

    print(f"\nDone. {ok} dataset(s). Run precompute next.")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    exit(main())
