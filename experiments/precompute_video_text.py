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

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


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

    if text_dim != 768:
        text_proj = torch.nn.Linear(text_dim, 768).to(dev)

    vision_dim = vision.output_dim if hasattr(vision, "output_dim") else 3584
    all_targets = [int(s.get("target", s.get("target_idx", s.get("label", 0)))) for s in samples]
    num_classes = max(all_targets) + 1 if all_targets else 8

    config = {
        "vision_dim": vision_dim,
        "audio_dim": 768,
        "num_classes": num_classes,
        "vision_encoder": vision_encoder,
        "text_encoder": text_encoder,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    for i, s in enumerate(samples):
        vid = s.get("video_path", s.get("video", s.get("video_id", "")))
        video_path = Path(vid) if Path(vid).is_absolute() else data_dir / vid
        if not video_path.exists():
            video_path = data_dir / "videos" / Path(vid).name
        text = str(s.get("text", s.get("question", s.get("caption", ""))))
        target = int(s.get("target", s.get("target_idx", s.get("label", 0))))

        frame = load_frame(video_path)
        if frame is None:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        if vision_encoder == "viscop":
            frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_t = frame_t.unsqueeze(0).to(dev)
        else:
            try:
                from mmfuse.preprocessing.preprocessor import VisionPreprocessor
            except ModuleNotFoundError:
                from preprocessing.preprocessor import VisionPreprocessor
            vprep = VisionPreprocessor(image_size=(224, 224))
            frame_t = vprep.preprocess(frame).unsqueeze(0).to(dev)

        with torch.no_grad():
            v_emb = vision(frame_t).squeeze(0)
        v1, v2 = v_emb.cpu(), v_emb.cpu()

        with torch.no_grad():
            t_emb = encode_text([text]).squeeze(0)
        if text_dim != 768:
            t_emb = text_proj(t_emb.unsqueeze(0)).squeeze(0).cpu()
        else:
            t_emb = t_emb.cpu()

        torch.save({"vision_camera1": v1, "vision_camera2": v2, "audio": t_emb, "target": target}, out_dir / f"{i:08d}.pt")
        if (i + 1) % 100 == 0:
            log.info("Precomputed %d / %d", i + 1, len(samples))

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
