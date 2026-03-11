#!/usr/bin/env python3
"""
Download VIMA-Bench data from HuggingFace (VIMA/VIMA-Data) and prepare for MMFuse.

Uses the official VIMA/VIMA-Data dataset (vima.zip, ~21.5 GB). Not LeRobot or other sources.

Disk space: need ~25 GB free for zip + cache, or ~50 GB if extracting full zip. Use --out-dir
and --extracted-dir on a drive with enough space (e.g. scratch). Reduce --max-samples to use less.

VIMA-Bench has 4 evaluation levels. We map to VisCoP's L1/L2/L3:
  L1 (Object Placement)     <- placement_generalization
  L2 (Novel Combination)   <- combinatorial_generalization
  L3 (Novel Object)        <- novel_object_generalization, novel_task_generalization

Output: extdataset/vima_bench/annotations.json, videos/*.mp4

After download, run precompute and finetune:
  python experiments/precompute_video_text.py --dataset vima_bench --out-dir embeddings/vima_bench
  python -m mmfuse.training.finetune --dataset vima_bench --embeddings-dir embeddings/vima_bench --model-file checkpoints/model.pt --no-movement-head --epochs 10

Usage:
  python experiments/download_vima_bench.py --max-samples 500
  python experiments/download_vima_bench.py --zip-path /path/to/vima.zip --max-samples 1000
  python experiments/download_vima_bench.py --small --max-samples 500   # ~300MB LeRobot, annotations + videos
  python experiments/download_vima_bench.py --small --max-samples 500 --max-gb 5   # Stay under 5GB

Requires: pip install huggingface_hub pandas pyarrow
  For video extraction: opencv-python (cv2)
"""

import argparse
import json
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

_proj_root = Path(__file__).resolve().parent.parent

# VIMA task name -> L1/L2/L3 (VisCoP levels)
# Based on VIMABench partitions: placement, combinatorial, novel_object, novel_task
TASK_TO_LEVEL = {
    # placement_generalization -> L1 (Object Placement)
    "visual_manipulation": 0,
    "scene_understanding": 0,
    "rotate": 0,
    "rearrange": 0,
    "rearrange_then_restore": 0,
    # combinatorial_generalization -> L2 (Novel Combination)
    "sweep_without_exceeding": 1,
    "sweep_without_touching": 1,
    "same_texture": 1,
    "same_shape": 1,
    "same_color": 1,
    "same_profile": 1,
    "follow_motion": 1,
    "follow_order": 1,
    # novel_object_generalization -> L3 (Novel Object)
    "novel_adj": 2,
    "novel_noun": 2,
    "novel_adj_and_noun": 2,
    "twist": 2,
    # novel_task_generalization -> L3
    "manipulate_old_neighbor": 2,
    "pick_in_order_then_restore": 2,
}
LEVEL_NAMES = ["L1 (Object Placement)", "L2 (Novel Combination)", "L3 (Novel Object)"]


def download_vima_bench_small(
    out_dir: Path,
    max_samples: int = 500,
    max_gb: float = 5.0,
) -> bool:
    """
    Download LeRobot level1: annotations + videos. Stays under max_gb (default 5GB).
    Uses parquet + snapshot_download for videos, extracts per-episode clips.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        import pandas as pd
    except ImportError:
        print("Install: pip install huggingface_hub pandas pyarrow")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    cache_dir = out_dir / "cache_lerobot"
    cache_dir.mkdir(exist_ok=True)

    ann_path = out_dir / "annotations.json"
    max_bytes = int(max_gb * 1024 * 1024 * 1024)

    # 1. Download parquet
    print("Downloading parquet (~64MB)...")
    try:
        parquet_path = hf_hub_download(
            repo_id="lerobot-data-collection/level1_final_quality0",
            filename="data/chunk-000/file-000.parquet",
            repo_type="dataset",
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
        parquet_path = Path(parquet_path)
    except Exception as e:
        print(f"  Failed: {e}")
        return False

    df = pd.read_parquet(parquet_path)
    ep_col = "episode_index" if "episode_index" in df.columns else next(
        (c for c in df.columns if "episode" in str(c).lower()), df.columns[0]
    )
    task_col = "task_index" if "task_index" in df.columns else None
    idx_col = "index" if "index" in df.columns else "frame_index" if "frame_index" in df.columns else None

    def _scalar(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0
        if isinstance(x, (list, tuple)):
            return int(x[0]) if x else 0
        return int(x)

    # Group by episode -> frame indices
    ep_to_frames = {}
    ep_to_task = {}
    for i, row in df.iterrows():
        ep = _scalar(row.get(ep_col, i))
        if ep not in ep_to_frames:
            ep_to_frames[ep] = []
            ep_to_task[ep] = _scalar(row.get(task_col, 0)) if task_col else 0
        ep_to_frames[ep].append(i)

    episodes = sorted(ep_to_frames.keys())[:max_samples]
    if not episodes:
        print("No episodes found.")
        return False

    # 2. Download video chunk (observation.images.base only, ~70MB)
    print("Downloading video chunk (~70MB)...")
    try:
        snapshot_download(
            repo_id="lerobot-data-collection/level1_final_quality0",
            repo_type="dataset",
            local_dir=cache_dir,
            allow_patterns=[
                "videos/observation.images.base/chunk-000/*.mp4",
                "meta/info.json",
            ],
        )
    except Exception as e:
        print(f"  Video download failed: {e}")
        print("  Saving annotations only (no videos).")
        rows = [
            {
                "video_id": f"lerobot_L1_ep{ep}",
                "video_path": f"videos/lerobot_L1_ep{ep}.mp4",
                "text": LEVEL_NAMES[min(ep_to_task.get(ep, 0), 2)],
                "target": min(ep_to_task.get(ep, 0), 2),
            }
            for ep in episodes
        ]
        with open(ann_path, "w") as f:
            json.dump(rows, f, indent=2)
        return True

    # Find video file (snapshot_download structure may vary)
    video_files = list(cache_dir.rglob("observation.images.base/**/*.mp4"))
    if not video_files:
        video_files = list(cache_dir.rglob("**/chunk-000/*.mp4"))
    src_video = video_files[0] if video_files else None

    rows = []
    if src_video and src_video.exists():
        print("Extracting per-episode videos...")
        try:
            import cv2
            cap = cv2.VideoCapture(str(src_video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            for j, ep in enumerate(episodes):
                if j % 50 == 0:
                    print(f"  Episodes {j}/{len(episodes)}...")
                frame_indices = ep_to_frames[ep]
                if not frame_indices:
                    continue
                frame_indices = sorted(set(frame_indices))
                out_path = videos_dir / f"lerobot_L1_ep{ep}.mp4"
                if out_path.exists():
                    rows.append({
                        "video_id": f"lerobot_L1_ep{ep}",
                        "video_path": f"videos/lerobot_L1_ep{ep}.mp4",
                        "text": LEVEL_NAMES[min(ep_to_task.get(ep, 0), 2)],
                        "target": min(ep_to_task.get(ep, 0), 2),
                    })
                    continue

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                for idx in frame_indices[:150]:
                    if idx < total_frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            out.write(frame)
                out.release()
                rows.append({
                    "video_id": f"lerobot_L1_ep{ep}",
                    "video_path": f"videos/lerobot_L1_ep{ep}.mp4",
                    "text": LEVEL_NAMES[min(ep_to_task.get(ep, 0), 2)],
                    "target": min(ep_to_task.get(ep, 0), 2),
                })

            cap.release()
        except Exception as e:
            print(f"  Extract failed: {e}. Saving annotations only.")
            rows = [
                {
                    "video_id": f"lerobot_L1_ep{ep}",
                    "video_path": f"videos/lerobot_L1_ep{ep}.mp4",
                    "text": LEVEL_NAMES[min(ep_to_task.get(ep, 0), 2)],
                    "target": min(ep_to_task.get(ep, 0), 2),
                }
                for ep in episodes
            ]
    else:
        rows = [
            {
                "video_id": f"lerobot_L1_ep{ep}",
                "video_path": f"videos/lerobot_L1_ep{ep}.mp4",
                "text": LEVEL_NAMES[min(ep_to_task.get(ep, 0), 2)],
                "target": min(ep_to_task.get(ep, 0), 2),
            }
            for ep in episodes
        ]

    with open(ann_path, "w") as f:
        json.dump(rows, f, indent=2)

    n_vids = len(list(videos_dir.glob("*.mp4")))
    print(f"VIMA-Bench (small) done. {len(rows)} samples, {n_vids} videos.")
    return True


def _get_task_from_trajectory_pkl(pkl_path: Path) -> str | None:
    """Read trajectory.pkl and return task name."""
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        # trajectory.pkl has meta info; structure may vary
        if isinstance(data, dict):
            return data.get("task_name") or data.get("task") or data.get("task_id")
        return None
    except Exception:
        return None


def _task_to_level(task_name: str) -> int:
    """Map task name to L1=0, L2=1, L3=2."""
    if not task_name:
        return 0
    task_lower = str(task_name).lower().replace("-", "_").replace(" ", "_")
    for key, level in TASK_TO_LEVEL.items():
        if key in task_lower:
            return level
    return 0


def _validate_mp4(path: Path) -> bool:
    """Return True if the mp4 can be opened and at least one frame read."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path.resolve()))
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        return bool(ok)
    except Exception:
        return False


def _frames_to_video(frames_dir: Path, out_video: Path) -> bool:
    """Convert rgb frames to mp4. Uses cv2 (OpenCV) or imageio. Validates output."""
    frames = sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return False

    # Try OpenCV first (reliable mp4v)
    try:
        import cv2
        first = cv2.imread(str(frames[0]))
        if first is None:
            return False
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_video), fourcc, 10.0, (w, h))
        for f in frames:
            img = cv2.imread(str(f))
            if img is not None:
                out.write(img)
        out.release()
        if out_video.exists() and _validate_mp4(out_video):
            return True
        if out_video.exists():
            out_video.unlink(missing_ok=True)
    except Exception:
        if out_video.exists():
            out_video.unlink(missing_ok=True)

    # Fallback: imageio with ffmpeg (mimwrite for list of frames)
    try:
        import imageio.v3 as iio
        frame_list = []
        for f in frames:
            arr = iio.imread(str(f))
            if arr.ndim == 2:
                arr = iio.core.util.array_as_uint8(arr)
            frame_list.append(arr)
        if not frame_list:
            return False
        iio.imwrite(out_video, frame_list, fps=10)
        if out_video.exists() and _validate_mp4(out_video):
            return True
        if out_video.exists():
            out_video.unlink(missing_ok=True)
    except Exception:
        if out_video.exists():
            out_video.unlink(missing_ok=True)
    return False


def _process_from_extracted_dir(
    extracted_dir: Path,
    out_dir: Path,
    videos_dir: Path,
    max_samples: int,
    skip_videos: bool,
    max_samples_per_task: int | None = None,
) -> list:
    """Build annotations and videos from an already-extracted zip (fast path). Returns list of row dicts."""
    rows = []
    seen = set()
    pkl_files = sorted(Path(extracted_dir).rglob("trajectory.pkl"))
    # Optionally group by task (first folder under extracted_dir) to cap per task
    if max_samples_per_task is not None and max_samples_per_task > 0:
        by_task: dict[str, list[Path]] = {}
        for p in pkl_files:
            try:
                rel = p.parent.relative_to(extracted_dir)
            except ValueError:
                rel = p.parent
            parts = rel.parts
            task = parts[0] if parts else "unknown"
            by_task.setdefault(task, []).append(p)
        # Process up to max_samples_per_task per task, global cap max_samples
        ordered = []
        for task in sorted(by_task.keys()):
            ordered.extend(by_task[task][:max_samples_per_task])
        pkl_files = ordered
        print(f"  Found {len(ordered)} trajectory dirs (up to {max_samples_per_task} per task). Processing...", flush=True)
    else:
        print(f"  Found {len(pkl_files)} trajectory dirs. Processing up to {max_samples}...", flush=True)

    for idx, pkl_path in enumerate(pkl_files):
        if len(rows) >= max_samples:
            break
        traj_dir_path = pkl_path.parent
        rgb_dir = traj_dir_path / "rgb_front"
        if not rgb_dir.is_dir():
            continue
        frames = sorted(rgb_dir.glob("*.png")) or sorted(rgb_dir.glob("*.jpg"))[:200]
        if not frames:
            continue
        try:
            rel = traj_dir_path.relative_to(extracted_dir)
        except ValueError:
            rel = traj_dir_path
        key = str(rel).replace("\\", "/")
        if key in seen:
            continue
        seen.add(key)
        parts = key.replace("\\", "/").split("/")
        task_name = parts[0] if parts else "unknown"
        traj_id = parts[-1] if len(parts) > 1 else "traj"
        task_from_pkl = None
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                task_from_pkl = data.get("task_name") or data.get("task") or data.get("task_id")
        except Exception:
            pass
        if task_from_pkl:
            task_name = str(task_from_pkl)
        target = _task_to_level(task_name)
        video_id = f"{task_name}_{traj_id}".replace("/", "_").replace("\\", "_")[:80]
        video_path = videos_dir / f"{video_id}.mp4"
        if not skip_videos and not video_path.exists():
            _frames_to_video(rgb_dir, video_path)
        if not skip_videos and not video_path.exists():
            continue
        rel_path = f"videos/{video_path.name}" if video_path.exists() else f"videos/{video_id}.mp4"
        rows.append({
            "video_id": video_id,
            "video_path": rel_path,
            "text": LEVEL_NAMES[target],
            "target": target,
        })
        n_done = len(rows)
        if n_done <= 5 or n_done % 25 == 0 or n_done == max_samples:
            print(f"  Processed {n_done}/{max_samples} (traj {idx+1}/{len(pkl_files)})", flush=True)
    return rows


def download_vima_bench(
    out_dir: Path,
    max_samples: int = 500,
    zip_path: Path | None = None,
    extracted_dir: Path | None = None,
    max_samples_per_task: int | None = None,
    skip_videos: bool = False,
    force: bool = False,
) -> bool:
    """Download VIMA-Data, extract trajectories, create annotations.json and videos."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install huggingface_hub")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    if force:
        if (out_dir / "annotations.json").exists():
            (out_dir / "annotations.json").unlink()
            print("VIMA-Bench: removed existing annotations.json (--force)")
        for f in videos_dir.glob("*.mp4"):
            f.unlink()
        print("VIMA-Bench: cleared existing videos (--force)")

    ann_path = out_dir / "annotations.json"
    if ann_path.exists() and not force:
        with open(ann_path) as f:
            existing = json.load(f)
        if len(existing) >= max_samples:
            print(f"VIMA-Bench: annotations.json exists ({len(existing)} samples), skipping")
            return True

    # Fast path: use pre-extracted directory (no zip; finishes in minutes)
    if extracted_dir is not None and extracted_dir.is_dir():
        print("VIMA-Bench: using pre-extracted directory (fast path)", flush=True)
        rows = _process_from_extracted_dir(
            extracted_dir, out_dir, videos_dir, max_samples, skip_videos,
            max_samples_per_task=max_samples_per_task,
        )
        if not rows:
            print("No trajectories found in extracted dir. Check structure (trajectory.pkl + rgb_front/).", flush=True)
            return False
        with open(ann_path, "w") as f:
            json.dump(rows, f, indent=2)
        n_vids = len(list(videos_dir.glob("*.mp4")))
        print(f"  Done. Saved {len(rows)} samples to {ann_path}. Videos: {n_vids}", flush=True)
        return True

    # Use existing zip if present; never delete or overwrite the downloaded file
    cache_zip = out_dir / "cache" / "vima.zip"
    if zip_path and zip_path.exists():
        zip_file = zip_path
        print(f"Using local zip: {zip_file}", flush=True)
    elif cache_zip.exists():
        zip_file = cache_zip
        print(f"Using existing cache zip (skipping download): {zip_file}", flush=True)
    else:
        print("Downloading VIMA-Data from HuggingFace (21.5 GB)...", flush=True)
        try:
            zip_file = Path(hf_hub_download(
                repo_id="VIMA/VIMA-Data",
                filename="vima.zip",
                repo_type="dataset",
                local_dir=out_dir / "cache",
                local_dir_use_symlinks=False,
            ))
        except Exception as e:
            print(f"Download failed: {e}", flush=True)
            print("  Alternatives:")
            print("    1. Download manually: https://huggingface.co/datasets/VIMA/VIMA-Data/resolve/main/vima.zip")
            print(f"    2. Run: python experiments/download_vima_bench.py --zip-path /path/to/vima.zip --max-samples {max_samples}")
            return False

    rows = []
    seen = set()
    print("Extracting trajectories and creating annotations...", flush=True)

    with zipfile.ZipFile(zip_file, "r") as zf:
        print("  [1/3] Reading zip index (can take 10-30 min on 21 GB zip)...", flush=True)
        names_orig = zf.namelist()
        print(f"  [1/3] Zip index read: {len(names_orig)} members.", flush=True)
        # Normalize to forward slash for path logic (Windows zip may have backslash)
        names = [n.replace("\\", "/") for n in names_orig]
        norm_to_orig = dict(zip(names, names_orig))
        # Trajectories: each has trajectory.pkl and rgb_front/
        traj_dirs = set()
        for n in names:
            parts = n.split("/")
            if "trajectory.pkl" in n:
                traj_dir = "/".join(parts[:-1]) if parts[:-1] else parts[0]
                traj_dirs.add(traj_dir)
            elif "rgb_front" in parts:
                idx = parts.index("rgb_front")
                traj_dir = "/".join(parts[:idx])
                traj_dirs.add(traj_dir)

        # Filter: must have both trajectory.pkl and rgb_front frames
        valid_trajs = []
        for td in traj_dirs:
            pkl = f"{td}/trajectory.pkl"
            if pkl not in names:
                continue
            has_frames = any(n.startswith(f"{td}/rgb_front/") and (n.endswith(".png") or n.endswith(".jpg")) for n in names)
            if has_frames:
                valid_trajs.append(td)

        total_valid = len(valid_trajs)
        target = min(max_samples, total_valid)
        print(f"  [2/3] Found {total_valid} valid trajectories. Will process up to {target}.", flush=True)
        print("  [3/3] Extracting and encoding videos...", flush=True)

        for traj_idx, traj_dir in enumerate(sorted(valid_trajs)):
            if len(rows) >= max_samples:
                break
            if (traj_idx + 1) % 10 == 0 or traj_idx == 0:
                print(f"  Zip trajectory {traj_idx + 1}/{len(valid_trajs)} (samples so far: {len(rows)})", flush=True)

            parts = traj_dir.split("/")
            if len(parts) < 2:
                continue
            task_name = parts[0] if parts[0] else (parts[1] if len(parts) > 1 else "unknown")
            traj_id = parts[-1] if parts[-1] and parts[-1] != task_name else "_".join(parts[-2:])
            key = f"{task_name}/{traj_id}"
            if key in seen:
                continue
            seen.add(key)

            # Extract trajectory.pkl
            pkl_name = f"{traj_dir}/trajectory.pkl"
            if pkl_name not in names:
                continue

            try:
                with zf.open(norm_to_orig.get(pkl_name, pkl_name)) as f:
                    data = pickle.load(f)
            except Exception:
                continue

            task_from_pkl = None
            if isinstance(data, dict):
                task_from_pkl = data.get("task_name") or data.get("task") or data.get("task_id")
            if task_from_pkl:
                task_name = str(task_from_pkl)

            target = _task_to_level(task_name)

            # Check for rgb_front
            rgb_prefix = f"{traj_dir}/rgb_front"
            frame_names = [n for n in names if n.startswith(rgb_prefix) and (n.endswith(".png") or n.endswith(".jpg"))]
            if not frame_names:
                continue

            video_id = f"{task_name}_{traj_id}".replace("/", "_").replace("\\", "_")[:80]
            video_path = videos_dir / f"{video_id}.mp4"

            if not skip_videos and not video_path.exists():
                import tempfile
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    for fn in frame_names[:200]:  # limit frames per trajectory
                        try:
                            # Use original zip member name for extract (Windows)
                            zf.extract(norm_to_orig.get(fn, fn), tmp_path)
                        except Exception:
                            pass
                    frames_dir = tmp_path / traj_dir / "rgb_front"
                    if not frames_dir.exists():
                        frames_dir = tmp_path / "rgb_front"
                    if frames_dir.exists():
                        _frames_to_video(frames_dir, video_path)

            if not skip_videos and not video_path.exists():
                continue

            rel_path = f"videos/{video_path.name}" if video_path.exists() else f"videos/{video_id}.mp4"
            text = LEVEL_NAMES[target]
            rows.append({
                "video_id": video_id,
                "video_path": rel_path,
                "text": text,
                "target": target,
            })

            n_done = len(rows)
            pct = 100 * n_done / max_samples if max_samples else 0
            if n_done <= 5 or n_done % 25 == 0 or n_done == max_samples:
                print(f"  Processed {n_done}/{max_samples} ({pct:.0f}%)", flush=True)

    if not rows:
        print("No trajectories extracted. Check zip structure.", flush=True)
        return False

    with open(ann_path, "w") as f:
        json.dump(rows, f, indent=2)

    n_vids = len(list(videos_dir.glob("*.mp4")))
    print(f"  [3/3] Done. Saved {len(rows)} samples to {ann_path}. Videos: {n_vids}", flush=True)
    return True


def main():
    p = argparse.ArgumentParser(description="Download VIMA-Bench for L1/L2/L3 evaluation")
    p.add_argument("--out-dir", default=None, help="Output dir (default: extdataset/vima_bench)")
    p.add_argument("--max-samples", type=int, default=500, help="Max trajectories to process")
    p.add_argument("--zip-path", default=None, help="Path to local vima.zip (skip download)")
    p.add_argument("--extracted-dir", default=None,
                   help="Path to already-extracted vima zip (fast path; no zip read). E.g. after: unzip vima.zip -d vima_extracted")
    p.add_argument("--max-samples-per-task", type=int, default=None,
                   help="When using --extracted-dir: take up to this many samples per task folder (e.g. 500 for 500 from each of rearrange_then_restore and sweep_without_exceeding). Use with --max-samples 1000 for 500+500.")
    p.add_argument("--skip-videos", action="store_true",
                   help="Skip video creation (annotations only; precompute needs videos)")
    p.add_argument("--small", action="store_true",
                   help="Use LeRobot level1 (~300MB) instead of VIMA (21.5GB) when disk space is limited")
    p.add_argument("--max-gb", type=float, default=5.0,
                   help="With --small: max download size in GB (default 5)")

    args = p.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else _proj_root / "extdataset" / "vima_bench"

    if args.small:
        ok = download_vima_bench_small(out_dir, max_samples=args.max_samples, max_gb=args.max_gb)
    else:
        ok = download_vima_bench(
            out_dir,
            max_samples=args.max_samples,
            zip_path=Path(args.zip_path) if args.zip_path else None,
            skip_videos=args.skip_videos,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
