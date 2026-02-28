#!/usr/bin/env python3
"""
Precompute embeddings for CALVIN ABC (manipulation) using CLIP (or VisCoP) vision.

Expected CALVIN layout under --data-root:
  data/    # parquet chunks with time-step rows
  video/   # video.image_base/, video.image_wrist/ with mp4s
  meta/    # episodes.jsonl, tasks.jsonl, etc. (not required here)

Parquet expectation (per row):
  - base_video: relative path to base camera mp4 under video/ (e.g. "chunk_0000/video.image_base/episode_0000.mp4")
  - wrist_video: relative path to wrist camera mp4 under video/ (e.g. "chunk_0000/video.image_wrist/episode_0000.mp4")
  - frame_index: int frame index within that mp4 (0-based)
  - task_index: int label for the task/action (0..N-1)

If your parquet schema differs, adapt the column names in load_samples().

Output:
  embeddings/calvin_abc_clip/
    config.json
    00000000.pt, 00000001.pt, ...

Each .pt file contains:
  - vision_camera1: CLIP embedding for base camera frame
  - vision_camera2: CLIP embedding for wrist camera frame
  - audio: zeros (no audio in CALVIN ABC)
  - text: zeros
  - target: task_index (int)

You can then run:
  python -m mmfuse.training.finetune_ckpt_only \
    --model-file checkpoints_clip_wav2vec_v3/ckpt_sdata_epoch_20.pt \
    --embeddings-dir embeddings/calvin_abc_clip \
    --output checkpoints_clip_wav2vec_v3/model_calvin_abc.pt \
    --no-movement-head --epochs 30
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import cv2

_proj_root = Path(__file__).resolve().parent.parent
import sys

if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from mmfuse.preprocessing.preprocessor import VisionPreprocessor  # type: ignore
from mmfuse.encoders.vision_encoder import VisionEncoder  # type: ignore
from mmfuse.encoders.vision_encoder_viscop import VisCoPVisionEncoder  # type: ignore
from config_modality import AUDIO_DIM, TEXT_DIM  # type: ignore


def load_frame_at(path: Path, frame_index: int) -> np.ndarray:
    """Load RGB frame at given index from mp4, or zeros if fail."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return np.zeros((224, 224, 3), dtype=np.uint8)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    idx = max(0, min(frame_index, n_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_samples(data_root: Path, max_rows: int | None = None):
    """
    Load CALVIN ABC rows from all parquet files under data/ into a list of samples.

    We derive video paths from the parquet chunk folder (e.g. chunk-000) and episode_index:
      base:  <chunk>/video.image_base/episode_{episode_index:06d}.mp4
      wrist: <chunk>/video.image_wrist/episode_{episode_index:06d}.mp4

    We also extract the continuous action vector from:
      - action.delta_ee_pos (3,)
      - action.delta_ee_rot (3,)
      - action.gripper     (1,)

    If your mp4 naming or action columns differ, adjust the patterns below.
    """
    data_dir = data_root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    samples: list[dict] = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        # Columns from InternData-Calvin_ABC
        required = [
            "frame_index",
            "episode_index",
            "task_index",
            "action.delta_ee_pos",
            "action.delta_ee_rot",
            "action.gripper",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Parquet {pf} missing required columns: {missing}")

        chunk_name = pf.parent.name
        for _, row in df.iterrows():
            episode_idx = int(row["episode_index"])
            frame_idx = int(row["frame_index"])
            task_idx = int(row["task_index"])

            # Build relative video paths
            base_rel = f"{chunk_name}/video.image_base/episode_{episode_idx:06d}.mp4"
            wrist_rel = f"{chunk_name}/video.image_wrist/episode_{episode_idx:06d}.mp4"

            # Build continuous action vector: [delta_pos(3), delta_rot(3), gripper(1)] -> (7,)
            dpos = np.array(row["action.delta_ee_pos"], dtype=np.float32).reshape(-1)
            drot = np.array(row["action.delta_ee_rot"], dtype=np.float32).reshape(-1)
            grip = np.array(row["action.gripper"], dtype=np.float32).reshape(-1)
            action_vec = np.concatenate([dpos, drot, grip], axis=0)

            samples.append(
                {
                    "base_video": base_rel,
                    "wrist_video": wrist_rel,
                    "frame_index": frame_idx,
                    "target": task_idx,
                    "action": action_vec,
                }
            )
            if max_rows is not None and len(samples) >= max_rows:
                return samples
    return samples


def precompute_calvin(
    data_root: Path,
    out_dir: Path,
    vision_encoder: str = "clip",
    device: str = "cuda",
    max_rows: int | None = None,
    batch_size: int = 32,
):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    samples = load_samples(data_root, max_rows=max_rows)
    if not samples:
        raise RuntimeError("No samples loaded from CALVIN ABC parquet files.")

    video_root = data_root / "video"
    vprep = VisionPreprocessor(image_size=(224, 224), normalize=True)

    if vision_encoder == "viscop":
        vision = VisCoPVisionEncoder(
            model_path="viscop_trained_models/viscop_qwen2.5_7b_viscop-lora_egocentric-expert",
            device=str(dev),
        ).to(dev)
        use_clip_preprocess = False
    else:
        vision = VisionEncoder(device=str(dev)).to(dev)
        use_clip_preprocess = True
    vision.eval()

    # Determine vision_dim from encoder
    dummy = torch.zeros(1, 3, 224, 224, device=dev)
    with torch.no_grad():
        dummy_emb = vision(dummy)
        if dummy_emb.dim() == 2:
            dummy_emb = dummy_emb.mean(dim=0, keepdim=True)
    vision_dim = int(dummy_emb.shape[-1])

    # Save config
    all_targets = [s["target"] for s in samples]
    num_classes = max(all_targets) + 1 if all_targets else 8
    config = {
        "vision_encoder": vision_encoder,
        "audio_encoder": "none",
        "vision_dim": vision_dim,
        "audio_dim": AUDIO_DIM,
        "num_classes": num_classes,
        "num_samples": len(samples),
        "cross_pair": False,
        "augment_variations": 1,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    n_batches = (len(samples) + batch_size - 1) // batch_size
    print(f"Precomputing CALVIN ABC: {len(samples)} samples in {n_batches} batches (batch_size={batch_size})")

    idx = 0
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        frames_base, frames_wrist, targets, actions = [], [], [], []
        for s in batch:
            base_path = video_root / s["base_video"]
            wrist_path = video_root / s["wrist_video"]
            frame_idx = s["frame_index"]
            f1 = load_frame_at(base_path, frame_idx)
            f2 = load_frame_at(wrist_path, frame_idx)
            if use_clip_preprocess:
                t1 = vprep.preprocess(f1)
                t2 = vprep.preprocess(f2)
            else:
                f1_resized = cv2.resize(f1, (224, 224))
                f2_resized = cv2.resize(f2, (224, 224))
                t1 = torch.from_numpy(f1_resized).permute(2, 0, 1).float() / 255.0
                t2 = torch.from_numpy(f2_resized).permute(2, 0, 1).float() / 255.0
            frames_base.append(t1)
            frames_wrist.append(t2)
            targets.append(s["target"])
            actions.append(s["action"])

        v1_batch = torch.stack(frames_base).to(dev)
        v2_batch = torch.stack(frames_wrist).to(dev)
        with torch.no_grad():
            emb1 = vision(v1_batch)
            emb2 = vision(v2_batch)
        if emb1.dim() == 2:
            emb1 = emb1
        if emb2.dim() == 2:
            emb2 = emb2
        emb1 = emb1.detach().cpu()
        emb2 = emb2.detach().cpu()

        for j in range(len(batch)):
            audio_emb = torch.zeros(AUDIO_DIM)
            text_emb = torch.zeros(TEXT_DIM)
            tgt = int(targets[j])
            act = torch.tensor(actions[j], dtype=torch.float32)
            torch.save(
                {
                    "vision_camera1": emb1[j],
                    "vision_camera2": emb2[j],
                    "audio": audio_emb,
                    "text": text_emb,
                    "target": tgt,
                    "action": act,
                },
                out_dir / f"{idx:08d}.pt",
            )
            idx += 1
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch)} / {len(samples)} samples")

    print(f"Done. Wrote {idx} embedding files to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=None, help="Root of CALVIN ABC (with data/, video/, meta/)")
    p.add_argument("--out-dir", type=str, required=True, help="Output dir for embeddings (e.g. embeddings/calvin_abc_clip)")
    p.add_argument("--vision-encoder", choices=["clip", "viscop"], default="clip")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap on number of rows for debugging")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    data_root = Path(args.data_root) if args.data_root else (_proj_root / "extdataset" / "calvin_abc")
    out_dir = Path(args.out_dir)
    precompute_calvin(
        data_root=data_root,
        out_dir=out_dir,
        vision_encoder=args.vision_encoder,
        device=args.device,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

