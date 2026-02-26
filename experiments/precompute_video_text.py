#!/usr/bin/env python3
"""
Precompute embeddings for video+text datasets (VideoMME, NeXTQA, EgoSchema, Charades).
Output format: vision_camera1, vision_camera2, audio (text→768dim), target.

Expected data structure in extdataset/<name>/:
  - videos/ or video files
  - annotations.json or annotations.csv with: video_id, text/question, target (class or MCQ index)

Usage:
  python experiments/precompute_video_text.py --dataset charades --out-dir embeddings/charades
  python experiments/precompute_video_text.py --dataset nextqa --out-dir embeddings/nextqa

Requires: vision encoder (viscop/clip), text encoder (viscop/CLIP/sentence-transformers).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import cv2

_proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj_root))

from config_modality import AUDIO_DIM, TEXT_DIM, get_embedding_config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def resolve_video_path(sample: dict, data_dir: Path) -> Path:
    """
    Resolve video file path from sample. Tries: data_dir/video_path, data_dir/videos/name,
    then recursive search under data_dir/videos for a file whose stem matches video_id.
    """
    vid = sample.get("video_path", sample.get("video", sample.get("video_id", "")))
    if not vid:
        return data_dir / "videos" / "missing.mp4"  # will not exist
    vid = str(vid).strip()
    data_dir = Path(data_dir)
    # Try explicit paths
    p = Path(vid)
    if p.is_absolute() and p.exists():
        return p
    candidate = data_dir / vid
    if candidate.exists():
        return candidate
    candidate = data_dir / "videos" / p.name
    if candidate.exists():
        return candidate
    # Recursive search: any file under data_dir/videos whose stem equals video_id
    video_id_stem = p.stem
    videos_dir = data_dir / "videos"
    if videos_dir.exists():
        for ext in (".mp4", ".mkv", ".avi", ".webm", ".mov", ""):
            for f in videos_dir.rglob(f"*{ext}" if ext else "*"):
                if f.is_file() and f.stem == video_id_stem:
                    return f
    return data_dir / vid  # return expected path for clearer error messages


def load_frame(video_path: Path, frame_idx: int = None) -> "np.ndarray":
    video_path = Path(video_path)
    if not video_path.exists():
        return None
    try:
        # Use CAP_FFMPEG to avoid OpenCV misinterpreting paths as image sequences
        cap = cv2.VideoCapture(str(video_path.resolve()), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        mid = frame_idx if frame_idx is not None else max(0, n // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None


def precompute_from_samples(
    samples: list,
    data_dir: Path,
    out_dir: Path,
    vision_encoder: str = "viscop",
    text_encoder: str = "clip",
    device: str = "cuda",
):
    """Run precompute on a list of samples. Each sample: {video_path, text, target}."""
    # Lazy load encoders
    try:
        from mmfuse.encoders.vision_encoder_viscop import VisCoPVisionEncoder
        from mmfuse.encoders.vision_encoder import VisionEncoder
    except ModuleNotFoundError:
        from encoders.vision_encoder_viscop import VisCoPVisionEncoder
        from encoders.vision_encoder import VisionEncoder

    dev = torch.device(device)
    if vision_encoder == "viscop":
        vision = VisCoPVisionEncoder(
            model_path="viscop_trained_models/viscop_qwen2.5_7b_viscop-lora_egocentric-expert",
            device=str(dev),
        ).to(dev)
    else:
        vision = VisionEncoder(device=str(dev)).to(dev)
    vision.eval()

    if text_encoder == "viscop":
        if vision_encoder != "viscop":
            raise ValueError("text_encoder=viscop requires vision_encoder=viscop (reuses same model)")
        tokenizer = getattr(vision.processor, "tokenizer", vision.processor)
        embed_layer = vision.model.get_input_embeddings()
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_len = min(512, getattr(vision.model.config, "max_position_embeddings", 2048))
        def encode_text(texts):
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=max_len,
                return_tensors="pt",
            ).to(dev)
            with torch.no_grad():
                emb = embed_layer(encoded.input_ids)
            mask = (encoded.input_ids != pad_id).unsqueeze(-1).float()
            if mask.sum() > 0:
                emb = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                emb = emb.mean(dim=1)
            return emb.float()
        text_dim = 3584
    elif text_encoder == "clip":
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        text_model = model.to(dev)
        text_model.eval()
        def encode_text(texts):
            t = tokenizer(texts).to(dev)
            with torch.no_grad():
                f = text_model.encode_text(t).float()
            return f / f.norm(dim=-1, keepdim=True)
        text_dim = 512
    else:
        try:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer("all-MiniLM-L6-v2").to(dev)
            def encode_text(texts):
                return torch.tensor(st.encode(texts, convert_to_numpy=True), device=dev).float()
            text_dim = 384
        except ImportError:
            raise RuntimeError("sentence-transformers required for --text-encoder bert")

    if text_dim != TEXT_DIM:
        text_proj = torch.nn.Linear(text_dim, TEXT_DIM).to(dev)

    vision_dim = vision.output_dim if hasattr(vision, "output_dim") else get_embedding_config(vision_encoder)["vision_dim"]
    all_targets = [int(s.get("target", s.get("target_idx", s.get("label", 0)))) for s in samples]
    num_classes = max(all_targets) + 1 if all_targets else 8

    config = {
        **get_embedding_config(vision_encoder),
        "vision_dim": vision_dim,
        "num_classes": num_classes,
        "vision_encoder": vision_encoder,
        "text_encoder": text_encoder,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Upfront check: how many video files actually exist?
    resolved_paths = [resolve_video_path(s, data_dir) for s in samples]
    n_exist = sum(1 for p in resolved_paths if p.exists())
    if n_exist == 0:
        log.warning(
            "No video files found under %s (checked %d samples). "
            "Put videos in e.g. %s with names matching annotations (e.g. <video_id>.mp4). "
            "NeXTQA: download videos separately (HuggingFace gives annotations only).",
            data_dir, len(samples), data_dir / "videos",
        )
        log.info("Example expected path for first sample: %s", resolved_paths[0] if resolved_paths else "N/A")
    else:
        log.info("Found %d / %d video files under %s", n_exist, len(samples), data_dir)

    n_failed_frames = 0
    n_nan_vision = 0
    for i, s in enumerate(samples):
        video_path = resolve_video_path(s, data_dir)
        text = str(s.get("text", s.get("question", s.get("caption", ""))))
        target = int(s.get("target", s.get("target_idx", s.get("label", 0))))

        #print(f"Video path: {video_path}")
        frame = load_frame(video_path)
        if frame is None:
            n_failed_frames += 1
            if n_failed_frames <= 5:
                log.warning("Frame load failed for %s (using zeros). Check path and codec.", video_path)
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            #print(f"Frame is None for {video_path}")

        if vision_encoder == "viscop":
            # Match SData: resize to 224x224 before VisCoP (SData does cv2.resize in build_embedding)
            frame = cv2.resize(frame, (224, 224))
            frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_t = frame_t.unsqueeze(0).to(dev)
            #print(f"Frame tensor: {frame_t}")
        else:
            try:
                from mmfuse.preprocessing.preprocessor import VisionPreprocessor
            except ModuleNotFoundError:
                from preprocessing.preprocessor import VisionPreprocessor
            vprep = VisionPreprocessor(image_size=(224, 224))
            frame_t = vprep.preprocess(frame).unsqueeze(0).to(dev)

        with torch.no_grad():
            v_emb = vision(frame_t).squeeze(0)
        if v_emb.dim() == 2:
            v_emb = v_emb.mean(dim=0)
        # Avoid saving NaN/Inf (e.g. VisCoP can produce NaN); use zeros so finetune gets valid inputs
        if not torch.isfinite(v_emb).all():
            n_nan_vision += 1
            if n_nan_vision <= 5:
                log.warning("Vision encoder produced NaN/Inf for sample %s (using zeros).", video_path)
            v_emb = torch.zeros_like(v_emb, dtype=v_emb.dtype, device=v_emb.device)
        v1, v2 = v_emb.cpu(), v_emb.cpu()

        if not text or not text.strip():
            t_emb = torch.zeros(TEXT_DIM, device=dev).cpu()
        else:
            with torch.no_grad():
                t_emb = encode_text([text]).squeeze(0)
            if text_dim != TEXT_DIM:
                t_emb = text_proj(t_emb.unsqueeze(0)).squeeze(0).cpu()
            else:
                t_emb = t_emb.cpu()

        audio_emb = torch.zeros(AUDIO_DIM, device=dev).cpu()
        torch.save({"vision_camera1": v1, "vision_camera2": v2, "audio": audio_emb, "text": t_emb, "target": target}, out_dir / f"{i:08d}.pt")
        if (i + 1) % 100 == 0:
            log.info("Precomputed %d / %d", i + 1, len(samples))

    if n_failed_frames:
        log.warning("Precompute used zero frames for %d / %d samples (vision embeddings may be zeros). Check video paths and OpenCV/ffmpeg.", n_failed_frames, len(samples))
    if n_nan_vision:
        log.warning(
            "Vision encoder produced NaN/Inf for %d / %d samples (saved as zeros). "
            "Check that VisCoP loaded correctly (see encoders/vision_encoder_viscop.py load_state_dict assign=True) and model is on GPU.",
            n_nan_vision, len(samples),
        )
    log.info("Done. Embeddings saved to %s", out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="video_mme, nextqa, charades, egoschema, vima_bench")
    p.add_argument("--data-dir", default=None, help="Override: extdataset/<dataset>")
    p.add_argument("--out-dir", required=True, help="Output embeddings dir")
    p.add_argument("--vision-encoder", choices=["viscop", "clip"], default="viscop")
    p.add_argument("--text-encoder", choices=["viscop", "clip", "bert"], default="viscop",
                   help="viscop: use VisCoP (requires vision_encoder=viscop)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _proj_root / "extdataset" / args.dataset
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "config.json").exists() and any(out_dir.glob("*.pt")):
        log.info("Embeddings already exist in %s. Skipping precompute.", out_dir)
        return 0

    if not data_dir.exists():
        log.error("Data dir not found: %s. Create extdataset/%s/ and add data.", data_dir, args.dataset)
        return 1

    # Dataset-specific loader: override in subclasses or pass annotations path
    annotations_path = data_dir / "annotations.json"
    if not annotations_path.exists():
        annotations_path = data_dir / "annotations.csv"
    if not annotations_path.exists():
        log.error("No annotations.json or annotations.csv in %s", data_dir)
        log.info("Expected format: list of {video_path, text, target} or CSV with video_path,text,target")
        return 1

    if annotations_path.suffix == ".json":
        with open(annotations_path) as f:
            samples = json.load(f)
    else:
        import csv
        samples = []
        with open(annotations_path) as f:
            for row in csv.DictReader(f):
                samples.append(row)

    log.info("Loaded %d samples from %s", len(samples), annotations_path)

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
