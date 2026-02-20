#!/usr/bin/env python3
"""
Download VIMA-Bench data from HuggingFace (VIMA/VIMA-Data) and prepare for MMFuse.

VIMA-Bench has 4 evaluation levels. We map to VisCoP's L1/L2/L3:
  L1 (Object Placement)     <- placement_generalization
  L2 (Novel Combination)    <- combinatorial_generalization
  L3 (Novel Object)         <- novel_object_generalization, novel_task_generalization

Output: extdataset/vima_bench/annotations.json, videos/*.mp4

After download, run precompute and evaluation:
  python experiments/precompute_video_text.py --dataset vima_bench --out-dir embeddings/vima_bench
  python experiments/run_dataset.py --dataset vima_bench --checkpoint checkpoints/ckpt_sdata_epoch_N.pt --linear-probe

Usage:
  python experiments/download_vima_bench.py --max-samples 500
  python experiments/download_vima_bench.py --zip-path /path/to/vima.zip --max-samples 1000

Requires: pip install huggingface_hub imageio imageio-ffmpeg
"""

import argparse
import json
import pickle
import zipfile
from pathlib import Path

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


def _frames_to_video(frames_dir: Path, out_video: Path) -> bool:
    """Convert rgb frames to mp4. Uses cv2 (OpenCV) or imageio."""
    frames = sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return False

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
        return out_video.exists()
    except Exception:
        pass

    try:
        import imageio.v3 as iio
        iio.imwrite(out_video, [iio.imread(str(f)) for f in frames], fps=10)
        return out_video.exists()
    except Exception:
        pass
    return False


def download_vima_bench(
    out_dir: Path,
    max_samples: int = 500,
    zip_path: Path | None = None,
    skip_videos: bool = False,
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

    ann_path = out_dir / "annotations.json"
    if ann_path.exists():
        with open(ann_path) as f:
            existing = json.load(f)
        if len(existing) >= max_samples:
            print(f"VIMA-Bench: annotations.json exists ({len(existing)} samples), skipping")
            return True

    # Download or use local zip
    if zip_path and zip_path.exists():
        zip_file = zip_path
        print(f"Using local zip: {zip_file}")
    else:
        print("Downloading VIMA-Data from HuggingFace (21.5 GB)...")
        try:
            zip_file = Path(hf_hub_download(
                repo_id="VIMA/VIMA-Data",
                filename="vima.zip",
                repo_type="dataset",
                local_dir=out_dir / "cache",
                local_dir_use_symlinks=False,
            ))
        except Exception as e:
            print(f"Download failed: {e}")
            print("  Alternatives:")
            print("    1. Download manually: https://huggingface.co/datasets/VIMA/VIMA-Data/resolve/main/vima.zip")
            print(f"    2. Run: python experiments/download_vima_bench.py --zip-path /path/to/vima.zip --max-samples {max_samples}")
            return False

    rows = []
    seen = set()
    print("Extracting trajectories and creating annotations...")

    with zipfile.ZipFile(zip_file, "r") as zf:
        names = zf.namelist()
        # Trajectories: each has trajectory.pkl and rgb_front/
        traj_dirs = set()
        for n in names:
            if "trajectory.pkl" in n:
                traj_dir = str(Path(n).parent).replace("\\", "/")
                traj_dirs.add(traj_dir)
            elif "rgb_front" in n:
                idx = n.split("/").index("rgb_front")
                traj_dir = "/".join(n.split("/")[:idx])
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

        for traj_dir in sorted(valid_trajs):
            if len(rows) >= max_samples:
                break

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
                with zf.open(pkl_name) as f:
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
                            zf.extract(fn, tmp_path)
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

            if len(rows) % 50 == 0 and rows:
                print(f"  Processed {len(rows)} trajectories...")

    if not rows:
        print("No trajectories extracted. Check zip structure.")
        return False

    with open(ann_path, "w") as f:
        json.dump(rows, f, indent=2)

    n_vids = len(list(videos_dir.glob("*.mp4")))
    print(f"VIMA-Bench done. Saved {len(rows)} samples to {ann_path}. Videos: {n_vids}")
    return True


def main():
    p = argparse.ArgumentParser(description="Download VIMA-Bench for L1/L2/L3 evaluation")
    p.add_argument("--out-dir", default=None, help="Output dir (default: extdataset/vima_bench)")
    p.add_argument("--max-samples", type=int, default=500, help="Max trajectories to process")
    p.add_argument("--zip-path", default=None, help="Path to local vima.zip (skip download)")
    p.add_argument("--skip-videos", action="store_true",
                   help="Skip video creation (annotations only; precompute needs videos)")

    args = p.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else _proj_root / "extdataset" / "vima_bench"

    ok = download_vima_bench(
        out_dir,
        max_samples=args.max_samples,
        zip_path=Path(args.zip_path) if args.zip_path else None,
        skip_videos=args.skip_videos,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
