#!/usr/bin/env python3
"""
Data loader for fine-tuning MMFuse on real-world datasets.
Supports: video_mme, nextqa, egoschema, charades, vima_bench, sdata.

Usage: source mmfuse-env/bin/activate && python training/finetune_dataset.py --dataset nextqa --max-samples 100
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_frame(video_path: Path, frame_idx: int = None) -> np.ndarray:
    """Load middle frame from video. Returns RGB (H,W,3) uint8."""
    video_path = Path(video_path)
    if not video_path.exists():
        return np.zeros((224, 224, 3), dtype=np.uint8)
    try:
        cap = cv2.VideoCapture(str(video_path.resolve()), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return np.zeros((224, 224, 3), dtype=np.uint8)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        mid = frame_idx if frame_idx is not None else max(0, n // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return np.zeros((224, 224, 3), dtype=np.uint8)


class FinetuneDataset(Dataset):
    """
    Loads from extdataset/<name>/annotations.json.
    Each sample: video(s), optional text, target.
    - Single video: video_path
    - Video pair (ego-exo): video_path, video_path_2
    - QA: text = question + options
    """

    def __init__(
        self,
        data_dir: Path,
        max_samples: int = None,
        image_size: tuple = (224, 224),
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.samples = []
        self._load_annotations(max_samples)

    def _load_annotations(self, max_samples: int):
        ann_path = self.data_dir / "annotations.json"
        if not ann_path.exists():
            ann_path = self.data_dir / "annotations.csv"
        if not ann_path.exists():
            raise FileNotFoundError(f"No annotations in {self.data_dir}")

        if ann_path.suffix == ".json":
            with open(ann_path) as f:
                data = json.load(f)
            rows = data if isinstance(data, list) else data.get("questions", [data])
        else:
            import csv
            rows = list(csv.DictReader(open(ann_path)))

        for i, row in enumerate(rows):
            if max_samples and i >= max_samples:
                break
            sample = self._parse_row(row)
            if sample:
                self.samples.append(sample)

    def _parse_row(self, row: dict) -> dict | None:
        """Parse row to {video_path, video_path_2?, text?, target}."""
        vid = row.get("video_id", row.get("video", ""))
        vp = row.get("video_path", f"videos/{vid}.mp4")
        if not Path(vp).is_absolute():
            vp = str(self.data_dir / vp) if vp.startswith("videos/") else str(self.data_dir / "videos" / f"{vid}.mp4")

        target = row.get("target")
        if target is None:
            ans = row.get("answer", "A")
            if isinstance(ans, int):
                target = ans
            else:
                target = ord(str(ans).strip().upper()[0]) - 65 if ans else 0
        target = int(target)

        text = None
        q = row.get("question", row.get("text", ""))
        opts = row.get("options", [])
        if not opts and any(k.startswith("a") and k[1:].isdigit() for k in row):
            opts = [row.get(f"a{i}", "") for i in range(5) if row.get(f"a{i}") is not None]
        if q:
            if isinstance(opts, (list, tuple)) and opts:
                text = str(q) + " " + " ".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts[:5]) if o)
            else:
                text = str(q)

        out = {"video_path": vp, "target": target}
        if text:
            out["text"] = text
        if "video_path_2" in row:
            out["video_path_2"] = str(self.data_dir / row["video_path_2"]) if not Path(row["video_path_2"]).is_absolute() else row["video_path_2"]
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frame = load_frame(s["video_path"])
        frame = cv2.resize(frame, self.image_size)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        out = {"frame": frame, "target": s["target"]}
        if "video_path_2" in s:
            frame2 = load_frame(s["video_path_2"])
            frame2 = cv2.resize(frame2, self.image_size)
            out["frame2"] = torch.from_numpy(frame2).permute(2, 0, 1).float() / 255.0
        if "text" in s:
            out["text"] = s["text"]
        return out


def collate_finetune(batch):
    """Collate batch. Frames stacked; text as list; targets stacked."""
    frames = torch.stack([b["frame"] for b in batch])
    targets = torch.tensor([b["target"] for b in batch], dtype=torch.long)
    out = {"frame": frames, "target": targets}
    if "frame2" in batch[0]:
        out["frame2"] = torch.stack([b["frame2"] for b in batch])
    if "text" in batch[0]:
        out["text"] = [b["text"] for b in batch]
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nextqa")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--max-samples", type=int, default=10)
    args = p.parse_args()

    proj = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else proj / "extdataset" / args.dataset
    ds = FinetuneDataset(data_dir, max_samples=args.max_samples)
    print(f"Dataset: {len(ds)} samples")
    if ds.samples:
        s = ds[0]
        print(f"  Sample keys: {list(s.keys())}")
        print(f"  Frame shape: {s['frame'].shape}, target: {s['target']}")
