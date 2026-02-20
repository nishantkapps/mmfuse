#!/usr/bin/env python3
"""
Download subsets of cross-dataset benchmarks for MMFuse experiments.
Creates extdataset/<name>/ with annotations and (where possible) minimal data.

Usage:
  python experiments/download_dataset_subsets.py video_mme --max-samples 100
  python experiments/download_dataset_subsets.py nextqa --max-samples 200
  python experiments/download_dataset_subsets.py all --max-samples 100

Requires: pip install datasets huggingface_hub
"""

import argparse
import ast
import json
import sys
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent


def download_video_mme(out_dir: Path, max_samples: int = 100):
    """Download VideoMME subset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("lmms-lab/Video-MME", split="test")
    n = min(max_samples, len(ds))
    rows = []
    for i in range(n):
        row = ds[i]
        vid = row.get("video_id", row.get("videoID", str(i)))
        q = row.get("question", "")
        opts = row.get("options", [])
        if hasattr(opts, "__iter__") and not isinstance(opts, str):
            opts = [str(o) if not isinstance(o, dict) else o.get("text", str(o)) for o in opts][:4]
        else:
            opts = []
        ans = str(row.get("answer", "A")).strip().upper()
        ans_map = {"A": 0, "B": 1, "C": 2, "D": 3}
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
    print(f"VideoMME: saved {len(rows)} samples to {out_dir}/annotations.json")
    print("  Note: Download videos separately from Video-MME (videos_chunked_*.zip)")
    return True


def download_nextqa(out_dir: Path, max_samples: int = 200):
    """Download NeXTQA subset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("lmms-lab/NExTQA", "MC", split="test")  # MC = multiple choice
    n = min(max_samples, len(ds))
    rows = []
    for i in range(n):
        row = ds[i]
        vid = row.get("video_id", row.get("vid", str(i)))
        q = row.get("question", "")
        a0 = row.get("a0", row.get("option_0", ""))
        a1 = row.get("a1", row.get("option_1", ""))
        a2 = row.get("a2", row.get("option_2", ""))
        a3 = row.get("a3", row.get("option_3", ""))
        a4 = row.get("a4", row.get("option_4", ""))
        ans = int(row.get("answer", row.get("correct", 0)))
        rows.append({
            "video_id": vid,
            "video_path": f"videos/{vid}.mp4",
            "question": q,
            "a0": a0, "a1": a1, "a2": a2, "a3": a3, "a4": a4,
            "answer": ans,
        })
    with open(out_dir / "annotations.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"NeXTQA: saved {len(rows)} samples to {out_dir}/annotations.json")
    print("  Note: Download videos from NExT-QA repo (https://github.com/doc-doc/NExT-QA)")
    return True


def download_charades(out_dir: Path, max_samples: int = 100):
    """Download Charades subset from HuggingFace (Aditya02/Charades-Action-Sequence-Sample)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset("Aditya02/Charades-Action-Sequence-Sample", split="train")
    except Exception as e:
        print(f"Charades not available: {e}")
        print("  Use original Charades from https://prior.allenai.org/projects/charades")
        return False
    n = min(max_samples, len(ds))
    rows = []
    for i in range(n):
        row = ds[i]
        vid = row.get("video_id", str(i))
        labels_raw = row.get("labels", [])
        if isinstance(labels_raw, str):
            try:
                labels_raw = ast.literal_eval(labels_raw) if labels_raw else []
            except Exception:
                labels_raw = []
        labels = list(labels_raw) if labels_raw else []
        target = int(labels[0]) if labels else 0
        text = row.get("script", "")
        if not text and row.get("descriptions"):
            desc = row["descriptions"]
            text = desc[0] if isinstance(desc, (list, tuple)) else str(desc)
        rows.append({
            "video_id": vid,
            "video_path": f"videos/{vid}.mp4",
            "text": text or "",
            "target": target,
        })
    with open(out_dir / "annotations.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Charades: saved {len(rows)} samples to {out_dir}/annotations.json")
    print("  Note: Videos are downloaded by HuggingFace to cache. Copy to extdataset/charades/videos/ if needed.")
    return True


def main():
    p = argparse.ArgumentParser(description="Download dataset subsets for MMFuse experiments")
    p.add_argument("dataset", choices=["video_mme", "nextqa", "charades", "egoschema", "vima_bench", "all"],
                   help="Dataset to download")
    p.add_argument("--max-samples", type=int, default=100, help="Max samples for subset (default: 100)")
    p.add_argument("--out-dir", default=None, help="Override output dir (default: extdataset/<name>)")
    args = p.parse_args()

    ext = _proj_root / "extdataset"
    ext.mkdir(exist_ok=True)

    datasets = ["video_mme", "nextqa", "charades"] if args.dataset == "all" else [args.dataset]

    ok = 0
    if "egoschema" in datasets:
        import subprocess
        import sys
        out = Path(args.out_dir) if args.out_dir else ext / "egoschema"
        r = subprocess.run(
            [sys.executable, str(_proj_root / "experiments" / "download_egoschema.py"),
             "--out-dir", str(out), "--max-samples", str(args.max_samples), "--skip-videos"],
            cwd=str(_proj_root),
        )
        if r.returncode == 0:
            ok += 1
        datasets = [d for d in datasets if d != "egoschema"]
    if "vima_bench" in datasets:
        print("VIMA-Bench: simulator-based. See experiments/DATASET_DOWNLOAD.md")
        datasets = [d for d in datasets if d != "vima_bench"]
    for name in datasets:
        out = Path(args.out_dir) if args.out_dir else ext / name
        if name == "video_mme":
            if download_video_mme(out, args.max_samples):
                ok += 1
        elif name == "nextqa":
            if download_nextqa(out, args.max_samples):
                ok += 1
        elif name == "charades":
            if download_charades(out, args.max_samples):
                ok += 1

    if ok == 0 and args.dataset != "all":
        print("No datasets downloaded. Check DATASET_DOWNLOAD.md for manual steps.")
        return 1
    print(f"\nDone. Downloaded {ok} dataset(s). Run precompute scripts next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
