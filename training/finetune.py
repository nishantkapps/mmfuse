#!/usr/bin/env python3
"""
Fine-tune MMFuse on real-world datasets (per-dataset, incremental).

HOW FINETUNE WORKS
------------------
Two data paths:

1) RAW VIDEOS (default)
   - Loads from extdataset/<dataset>/ (annotations.json + videos/*.mp4).
   - Each batch: load middle frame from video -> run through vision encoder (VisCoP/CLIP) -> fusion -> action head.
   - No precompute step; VisCoP runs on-the-fly. For vima_bench you must run
     experiments/download_vima_bench.py first to create extdataset/vima_bench/.
   - Risk: VisCoP can output NaN for some inputs -> loss NaN -> all batches skipped.

2) PRECOMPUTED EMBEDDINGS (recommended for vima_bench)
   - Run precompute first, then finetune on .pt files (no vision encoder in the loop).
   - Step 1: python experiments/download_vima_bench.py --max-samples 500
   - Step 2: python experiments/precompute_video_text.py --dataset vima_bench --out-dir embeddings/vima_bench
   - Step 3: python -m mmfuse.training.finetune --dataset vima_bench --embeddings-dir embeddings/vima_bench --model-file checkpoints/model.pt --no-movement-head --epochs 10
   - This avoids NaN from VisCoP and matches how SData training works (train on precomputed embeddings).

Usage (raw):  python -m mmfuse.training.finetune --dataset nextqa --max-samples 500 --epochs 5
Usage (precomputed): python -m mmfuse.training.finetune --dataset vima_bench --embeddings-dir embeddings/vima_bench --model-file checkpoints/model.pt --epochs 10
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Ensure mmfuse package
_proj = Path(__file__).resolve().parent.parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

try:
    from mmfuse.training.finetune_dataset import (
        FinetuneDataset,
        PrecomputedFinetuneDataset,
        collate_finetune,
        collate_precomputed,
    )
    from mmfuse.training.finetune_model import MMFuseFinetuneModel
    from config_modality import (
        FUSION_DIM,
        AUDIO_DIM,
        TEXT_DIM,
        PRESSURE_DIM,
        EMG_DIM,
        get_modality_dims,
    )
except ImportError:
    from finetune_dataset import (
        FinetuneDataset,
        PrecomputedFinetuneDataset,
        collate_finetune,
        collate_precomputed,
    )
    from finetune_model import MMFuseFinetuneModel
    try:
        from config_modality import (
            FUSION_DIM,
            AUDIO_DIM,
            TEXT_DIM,
            PRESSURE_DIM,
            EMG_DIM,
            get_modality_dims,
        )
    except ImportError:
        import sys
        sys.path.insert(0, str(_proj))
        from config_modality import (
            FUSION_DIM,
            AUDIO_DIM,
            TEXT_DIM,
            PRESSURE_DIM,
            EMG_DIM,
            get_modality_dims,
        )

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _unfreeze_last_n_layers(encoder_module, n: int, encoder_kind: str):
    """Unfreeze only the last n transformer layers. encoder_kind: 'viscop' | 'clip'. Returns count unfrozen."""
    if n <= 0:
        return 0
    for p in encoder_module.parameters():
        p.requires_grad = False
    root = getattr(encoder_module, "model", encoder_module)
    layers = None
    if encoder_kind == "viscop":
        if hasattr(root, "model") and hasattr(root.model, "model") and hasattr(root.model.model, "layers"):
            layers = root.model.model.layers
        elif hasattr(root, "model") and hasattr(root.model, "layers"):
            layers = root.model.layers
        elif hasattr(root, "layers"):
            layers = root.layers
    elif encoder_kind == "clip":
        if hasattr(root, "visual") and hasattr(root.visual, "transformer"):
            layers = getattr(root.visual.transformer, "resblocks", None) or getattr(root.visual.transformer, "layers", None)
        elif hasattr(root, "transformer"):
            layers = getattr(root.transformer, "resblocks", None) or getattr(root.transformer, "layers", None)
    if layers is None or not isinstance(layers, nn.ModuleList):
        log.warning("Unfreeze last %d layers: no layer list for %s", n, encoder_kind)
        return 0
    total = len(layers)
    take = min(n, total)
    for i in range(total - take, total):
        for p in layers[i].parameters():
            p.requires_grad = True
    log.info("Unfroze last %d layer(s) of vision (%s), total %d", take, encoder_kind, total)
    return take


def get_text_encoder_module(device, text_encoder="clip"):
    """Return (text encoder module, output_dim). Module is trainable."""
    if text_encoder == "clip":
        import open_clip
        full_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        # Use text tower only; wrap for batch encoding
        class CLIPTextEncoder(nn.Module):
            def __init__(self, clip_model, tokenizer, device):
                super().__init__()
                self.model = clip_model
                self.tokenizer = tokenizer
                self.device = device

            def forward(self, texts):
                t = self.tokenizer(texts).to(self.device)
                return self.model.encode_text(t).float()

        enc = CLIPTextEncoder(full_model, tokenizer, device).to(device)
        return enc, 512
    try:
        from mmfuse.encoders.text_encoder import TextEncoder
        enc = TextEncoder(output_dim=512, device=str(device))
        return enc, 512
    except Exception:
        return None, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nextqa",
                   help="Dataset name (e.g. nextqa -> extdataset/nextqa) or path to data dir (e.g. mmfuse/extdataset/nextqa). Paths are relative to cwd.")
    p.add_argument("--data-dir", default=None, help="Override: explicit path to raw data dir (relative to cwd). Overrides --dataset path.")
    p.add_argument("--embeddings-dir", default=None,
                   help="Use precomputed embeddings from this dir (run precompute_video_text.py first). Avoids running vision encoder in the loop and prevents NaN loss.")
    p.add_argument("--max-samples", type=int, default=500,
                   help="Max samples to load from dataset (default 500). Use a large value (e.g. 10000) or run without limit in finetune_dataset to use all videos.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--model-file", default=None,
                   help="Canonical model path: load from here (if exists) and save back here after finetuning. Use the same path for all finetuning runs so the file gets updated each time.")
    p.add_argument("--checkpoint", default=None, help="Resume from checkpoint (alternative to --model-file; save goes to out-dir)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vision-encoder", default="viscop", choices=["clip", "viscop"],
                   help="Vision encoder (default: viscop to match SData-trained model)")
    p.add_argument("--text-encoder", default="clip", help="clip or bert (mmfuse TextEncoder)")
    p.add_argument("--no-movement-head", action="store_true", help="Disable movement head (e.g. QA datasets)")
    p.add_argument("--freeze-vision", action="store_true", help="Keep vision encoder frozen (default: unfreeze for fine-tuning)")
    p.add_argument("--unfreeze-vision-layers", type=int, default=2, metavar="N",
                   help="Unfreeze only last N vision encoder layers (0 = unfreeze all). Default 2.")
    p.add_argument("--freeze-text", action="store_true", help="Keep text encoder frozen (default).")
    p.add_argument("--unfreeze-text", action="store_true", help="Train text encoder (default: frozen to save memory).")
    p.add_argument("--vision-lr", type=float, default=1e-5, help="LR for vision encoder (smaller than main)")
    p.add_argument("--text-lr", type=float, default=1e-5, help="LR for text encoder")
    p.add_argument("--train-answer-head-only", action="store_true",
                   help="Freeze entire model (fusion + encoders); train only the answer/action head (e.g. for NextQA).")
    args = p.parse_args()
    # Text encoder frozen by default (--unfreeze-text to train it)
    args.freeze_text = not args.unfreeze_text

    device = torch.device(args.device)
    # Relative paths are resolved from cwd (where you run the command), not script location
    cwd = Path.cwd()
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = cwd / data_dir
        dataset_name = data_dir.name
    elif "/" in args.dataset or "\\" in args.dataset:
        # --dataset is a path (e.g. mmfuse/extdataset/nextqa)
        data_dir = Path(args.dataset)
        if not data_dir.is_absolute():
            data_dir = cwd / data_dir
        dataset_name = data_dir.name
    else:
        data_dir = cwd / "extdataset" / args.dataset
        dataset_name = args.dataset
    out_dir = Path(args.out_dir) if args.out_dir else cwd / "checkpoints" / f"finetune_{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_file = Path(args.model_file) if args.model_file else None
    if model_file is not None and not model_file.is_absolute():
        model_file = cwd / model_file
    if model_file is not None and model_file.exists():
        load_path = model_file
    elif args.checkpoint and Path(args.checkpoint).exists():
        load_path = Path(args.checkpoint) if Path(args.checkpoint).is_absolute() else cwd / args.checkpoint
    else:
        load_path = None
    if model_file and not model_file.exists():
        log.info("Model file %s does not exist yet; will save to it after training", model_file)

    # fusion_dim: from checkpoint when loading, else central config
    fusion_dim = FUSION_DIM
    if load_path and load_path.exists():
        ckpt_pre = torch.load(load_path, map_location="cpu", weights_only=True)
        fusion_dim = ckpt_pre.get("fusion_dim", FUSION_DIM)
        log.info("Using fusion_dim=%d from %s", fusion_dim, load_path)

    use_precomputed = bool(args.embeddings_dir)
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else None
    if embeddings_dir is not None and not embeddings_dir.is_absolute():
        embeddings_dir = cwd / embeddings_dir

    if use_precomputed:
        embeddings_dir = embeddings_dir.resolve()
        if not embeddings_dir.exists():
            log.error("Embeddings dir not found: %s. Run precompute first, e.g.: python experiments/precompute_video_text.py --dataset %s --out-dir %s",
                      embeddings_dir, dataset_name, embeddings_dir)
            return 1
        ds = PrecomputedFinetuneDataset(embeddings_dir, max_samples=args.max_samples)
        if len(ds) == 0:
            log.error("No .pt files in %s", embeddings_dir)
            return 1
        n_class = ds.config.get("num_classes", max(ds[i]["target"] for i in range(len(ds))) + 1)
        has_text = False
        log.info("Precomputed dataset: %s (%d samples, %d classes)", embeddings_dir, len(ds), n_class)
        # Log first-file vision stats so we see exactly what is being loaded from this dir
        pt_files = sorted(embeddings_dir.glob("*.pt"))
        first_pt = pt_files[0] if pt_files else None
        if first_pt:
            sample = torch.load(first_pt, map_location="cpu", weights_only=True)
            v1 = sample.get("vision_camera1")
            if v1 is not None and isinstance(v1, torch.Tensor):
                log.info("First file %s: vision_camera1 min=%.4f max=%.4f", first_pt.name, float(v1.min()), float(v1.max()))
            else:
                log.info("First file %s: no vision_camera1 key", first_pt.name)
        log.warning("Precomputed mode: only fusion + action head are trained. Vision/text encoders are NOT loaded. To train last N vision layers + fusion on raw videos, run WITHOUT --embeddings-dir (and put videos in extdataset/<dataset>/).")
    else:
        log.info("Loading dataset %s from %s (raw videos)", dataset_name, data_dir)
        ds = FinetuneDataset(data_dir, max_samples=args.max_samples)
        if len(ds) == 0:
            log.error("No samples found in %s. For vima_bench run: python experiments/download_vima_bench.py --max-samples 500", data_dir)
            return 1
        n_class = max(s["target"] for s in ds.samples) + 1
        has_text = "text" in ds.samples[0]
        log.info("Samples: %d, classes: %d, has_text: %s", len(ds), n_class, has_text)

    train_size = int(0.9 * len(ds))
    train_ds, val_ds = random_split(ds, [train_size, len(ds) - train_size])
    collate_fn = collate_precomputed if use_precomputed else collate_finetune
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    if not use_precomputed:
        batches_per_epoch = len(train_dl)
        log.info("Raw training: %d train samples, %d batches/epoch (batch_size=%d). Use --max-samples N to use more than %d samples.",
                 train_size, batches_per_epoch, args.batch_size, len(ds))

    # Modality dims: central config (config_modality.py)
    load_from_sdata = load_path is not None and load_path.exists()
    if use_precomputed:
        vision_encoder_pre = ds.config.get("vision_encoder", "clip")
        modality_dims = get_modality_dims(vision_encoder_pre)
    elif load_from_sdata:
        modality_dims = get_modality_dims(args.vision_encoder)
    else:
        # Raw only, no checkpoint: minimal modalities (vision + optional text)
        modality_dims = {"vision_camera1": get_modality_dims(args.vision_encoder)["vision_camera1"]}
        if has_text:
            _, text_dim = get_text_encoder_module(device, args.text_encoder)
            modality_dims["text"] = text_dim

    model = MMFuseFinetuneModel(
        modality_dims=modality_dims,
        fusion_dim=fusion_dim,
        num_classes=n_class,
        use_movement_head=not args.no_movement_head and dataset_name == "sdata",
    ).to(device)

    # Step 1 & 5: When --train-answer-head-only (e.g. NextQA), freeze entire model except answer/action head
    train_answer_head_only = getattr(args, "train_answer_head_only", False)
    if train_answer_head_only:
        for p in model.fusion.parameters():
            p.requires_grad = False
        if model.movement_head is not None:
            for p in model.movement_head.parameters():
                p.requires_grad = False
        for p in model.action_head.parameters():
            p.requires_grad = True
        log.info("NextQA/answer-head-only mode: frozen fusion + movement; training only action (answer) head.")

    # Vision and text encoders only when using raw videos (not precomputed)
    vision = None
    text_encoder_module = None
    unfreeze_vis_n = max(0, getattr(args, "unfreeze_vision_layers", 0))
    if not use_precomputed:
        freeze_vision = args.freeze_vision or (unfreeze_vis_n > 0)
        if args.vision_encoder == "viscop":
            from mmfuse.encoders.vision_encoder_viscop import VisCoPVisionEncoder
            vision = VisCoPVisionEncoder(
                model_path="viscop_trained_models/viscop_qwen2.5_7b_viscop-lora_egocentric-expert",
                device=str(device),
                frozen=freeze_vision,
            )
        else:
            from mmfuse.encoders.vision_encoder import VisionEncoder
            vision = VisionEncoder(device=str(device), frozen=freeze_vision)
        vision = vision.to(device)
        if unfreeze_vis_n > 0:
            _unfreeze_last_n_layers(vision, unfreeze_vis_n, "viscop" if args.vision_encoder == "viscop" else "clip")
            vision.train()
            log.info("Vision encoder: last %d layer(s) trainable (lr=%s)", unfreeze_vis_n, args.vision_lr)
        elif args.freeze_vision:
            for p in vision.parameters():
                p.requires_grad = False
            vision.eval()
            log.info("Vision encoder: frozen")
        else:
            vision.train()
            log.info("Vision encoder: trainable (lr=%s)", args.vision_lr)

        if has_text:
            text_encoder_module, _ = get_text_encoder_module(device, args.text_encoder)
            text_encoder_module = text_encoder_module.to(device)
            if args.freeze_text:
                for p in text_encoder_module.parameters():
                    p.requires_grad = False
                text_encoder_module.eval()
                log.info("Text encoder: frozen")
            else:
                text_encoder_module.train()
                log.info("Text encoder: trainable (lr=%s)", args.text_lr)
    else:
        log.info("Using precomputed embeddings (no vision/text encoder in loop)")

    # Optimizer: train only answer head when --train-answer-head-only; else fusion + heads + optional vision/text
    if train_answer_head_only:
        params = [{"params": model.action_head.parameters(), "lr": args.lr}]
        if vision is not None:
            for p in vision.parameters():
                p.requires_grad = False
        if text_encoder_module is not None:
            for p in text_encoder_module.parameters():
                p.requires_grad = False
    else:
        params = [{"params": model.parameters(), "lr": args.lr}]
        if vision is not None and (not args.freeze_vision or unfreeze_vis_n > 0):
            params.append({"params": vision.parameters(), "lr": args.vision_lr})
        if text_encoder_module is not None:
            params.append({"params": text_encoder_module.parameters(), "lr": args.text_lr})
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Load from checkpoint/model-file for incremental fine-tuning (load to CPU to avoid OOM, then load_state_dict copies to model on device)
    if load_path and load_path.exists():
        ckpt = torch.load(load_path, map_location="cpu", weights_only=True)
        if "fusion_state" in ckpt:
            fusion_ckpt = ckpt["fusion_state"]
            fusion_model_state = model.fusion.state_dict()
            # Only load keys with matching shapes (e.g. precomputed CLIP 512-dim vs checkpoint VisCoP 3584-dim)
            fusion_filtered = {k: v for k, v in fusion_ckpt.items() if k in fusion_model_state and fusion_model_state[k].shape == v.shape}
            if len(fusion_filtered) < len(fusion_ckpt):
                skipped = set(fusion_ckpt) - set(fusion_filtered)
                log.warning("Fusion: skipped %d weights with shape mismatch (precomputed dim may differ from checkpoint): %s", len(skipped), sorted(skipped)[:5])
            model.fusion.load_state_dict(fusion_filtered, strict=False)
            log.info("Loaded fusion from %s", load_path)
        if "model_state" in ckpt and ckpt.get("num_classes") == n_class:
            model.action_head.load_state_dict(ckpt["model_state"], strict=False)
            log.info("Loaded action head from %s", load_path)
        if "movement_state" in ckpt and model.movement_head is not None:
            model.movement_head.load_state_dict(ckpt["movement_state"], strict=False)
            log.info("Loaded movement head from %s", load_path)
        if "vision_state" in ckpt and vision is not None and (not args.freeze_vision or unfreeze_vis_n > 0):
            vision.load_state_dict(ckpt["vision_state"], strict=False)
            log.info("Loaded vision encoder from %s", load_path)
        if "text_state" in ckpt and text_encoder_module is not None and not args.freeze_text:
            text_encoder_module.load_state_dict(ckpt["text_state"], strict=False)
            log.info("Loaded text encoder from %s", load_path)

    # CLIP normalization (from open_clip) for raw-video path when using CLIP vision
    clip_mean = clip_std = None
    if not use_precomputed and args.vision_encoder != "viscop":
        from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
        clip_mean = torch.tensor(OPENAI_DATASET_MEAN, device=device).view(1, 3, 1, 1)
        clip_std = torch.tensor(OPENAI_DATASET_STD, device=device).view(1, 3, 1, 1)

    # Model architecture and parameter counts (like train_sdata)
    print("\n--- Finetune model (fusion + action_head + movement_head)")
    print(model)
    def _count(m):
        return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)
    components = [("fusion + heads", model)]
    if vision is not None:
        components.append((f"vision ({args.vision_encoder})", vision))
    if text_encoder_module is not None:
        components.append(("text_encoder", text_encoder_module))
    print("\n--- Parameter counts")
    if use_precomputed:
        print("  vision, text: from precomputed embeddings (no encoder in loop)")
    total_all, trainable_all = 0, 0
    for name, m in components:
        t, tr = _count(m)
        total_all += t
        trainable_all += tr
        status = "trainable" if tr > 0 else "frozen"
        print(f"  {name}: {t:,} ({status})")
    log.info("TOTAL: %s | Trainable: %s", f"{total_all:,}", f"{trainable_all:,}")
    for mod_name, mod in components:
        trainable = [(n, p.shape) for n, p in mod.named_parameters() if p.requires_grad]
        if trainable:
            log.info("Trainable in %s: %s", mod_name, [(n, tuple(s)) for n, s in trainable])
    log.info("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        if vision is not None and not args.freeze_vision:
            print("Training vision encoder...")
            vision.train()
        if text_encoder_module is not None and not args.freeze_text:
            print("Training text encoder...")
            text_encoder_module.train()
        total_loss, correct, total, n_steps = 0.0, 0, 0, 0
        for batch_idx, batch in enumerate(train_dl):
            targets = batch["target"].to(device)

            if use_precomputed:
                # Embeddings from precomputed .pt files; add missing modalities as zeros for SData fusion (6 modalities)
                B = targets.shape[0]
                emb = {k: batch[k].to(device).float() for k in batch if k != "target" and isinstance(batch[k], torch.Tensor)}
                if "pressure" not in emb:
                    emb["pressure"] = torch.zeros(B, PRESSURE_DIM, device=device, dtype=torch.float32)
                if "emg" not in emb:
                    emb["emg"] = torch.zeros(B, EMG_DIM, device=device, dtype=torch.float32)
                if "text" not in emb:
                    emb["text"] = torch.zeros(B, TEXT_DIM, device=device, dtype=torch.float32)
                # NextQA has no audio; use zeros for audio, EMG, and pressure (ignored modalities)
                if "audio" not in emb:
                    emb["audio"] = torch.zeros(B, AUDIO_DIM, device=device, dtype=torch.float32)
                for k in emb:
                    emb[k] = torch.where(torch.isfinite(emb[k]), emb[k], torch.zeros_like(emb[k]))
                # Clamp to avoid overflow in fusion (Linear/BN can blow up with large precomputed values)
                for k in emb:
                    emb[k] = emb[k].clamp(-30.0, 30.0)
                if batch_idx == 0 and epoch == 0:
                    stats = {k: (float(emb[k].min()), float(emb[k].max()), bool(torch.isnan(emb[k]).any())) for k in emb}
                    log.info("Precomputed first batch emb stats: %s", stats)
                    v1, v2 = emb.get("vision_camera1"), emb.get("vision_camera2")
                    if v1 is not None and v2 is not None and v1.numel() and float(v1.abs().max()) < 1e-7 and float(v2.abs().max()) < 1e-7:
                        log.warning("Precomputed vision embeddings are near zero in this batch. Loaded from: %s. Training will be trivial without vision.", embeddings_dir)
            else:
                frames = batch["frame"].to(device).float()
                if args.vision_encoder == "viscop":
                    frames_in = frames.clamp(0.0, 1.0)
                else:
                    frames_in = (frames.clamp(0.0, 1.0) - clip_mean) / clip_std
                v_emb = vision(frames_in)
                if v_emb.dim() == 3:
                    v_emb = v_emb.mean(dim=1)
                v_emb = torch.where(torch.isfinite(v_emb), v_emb, torch.zeros_like(v_emb))
                B = v_emb.shape[0]
                emb = {"vision_camera1": v_emb}
                if load_from_sdata:
                    emb["vision_camera2"] = torch.zeros_like(v_emb, device=device)
                    emb["audio"] = torch.zeros(B, AUDIO_DIM, device=device, dtype=v_emb.dtype)
                    emb["text"] = torch.zeros(B, TEXT_DIM, device=device, dtype=v_emb.dtype)
                    emb["pressure"] = torch.zeros(B, PRESSURE_DIM, device=device, dtype=v_emb.dtype)
                    emb["emg"] = torch.zeros(B, EMG_DIM, device=device, dtype=v_emb.dtype)
                if has_text and "text" in batch and text_encoder_module is not None:
                    txt_emb = text_encoder_module(batch["text"])
                    if load_from_sdata and txt_emb.shape[-1] != TEXT_DIM:
                        pad = TEXT_DIM - txt_emb.shape[-1]
                        if pad > 0:
                            txt_emb = torch.cat([txt_emb, torch.zeros(txt_emb.shape[0], pad, device=device, dtype=txt_emb.dtype)], dim=-1)
                    emb["text"] = txt_emb
                for k in emb:
                    emb[k] = torch.where(torch.isfinite(emb[k]), emb[k], torch.zeros_like(emb[k]))

                if batch_idx == 0 and epoch == 0:
                    log.info("First batch: frame shape %s, v_emb min=%.4f max=%.4f has_nan=%s",
                             tuple(frames.shape), v_emb.min().item(), v_emb.max().item(), bool(torch.isnan(v_emb).any()))

            logits, mov, kl = model(emb, return_kl=True)
            loss = criterion(logits, targets)
            if kl:
                kl_sum = sum(v for v in kl.values() if torch.isfinite(v).all())
                if isinstance(kl_sum, torch.Tensor):
                    loss = loss + 0.1 * kl_sum
            if not torch.isfinite(loss):
                if batch_idx == 0 and epoch == 0:
                    log.warning("First-batch loss is nan/inf. Try --embeddings-dir (precompute first) or --vision-encoder clip.")
                    _loss_val = loss.item() if isinstance(loss, torch.Tensor) and torch.isfinite(loss) else float("nan")
                    log.warning("Diagnostic: logits has_nan=%s has_inf=%s; loss=%s",
                                bool(torch.isnan(logits).any()), bool(torch.isinf(logits).any()), _loss_val)
                log.warning("Skipping batch with non-finite loss (nan/inf)")
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_steps += 1
            total_loss += loss.item()
            correct += (logits.argmax(1) == targets).sum().item()
            total += len(targets)

        acc = correct / total if total else 0
        avg_loss = total_loss / n_steps if n_steps else float("nan")
        total_batches = len(train_dl)
        log.info("Epoch %d/%d loss=%.4f acc=%.2f%% | batches updated %d/%d",
                 epoch + 1, args.epochs, avg_loss, 100 * acc, n_steps, total_batches)
        if n_steps == 0 and total_batches > 0:
            log.warning("No batches were updated (loss was nan/inf every time). Try --vision-encoder clip or check data.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if n_steps == 0 and epoch == 0:
            log.error("No batches had finite loss. Use precomputed embeddings: 1) precompute_video_text.py --dataset %s --out-dir embeddings/%s  2) finetune --embeddings-dir embeddings/%s",
                      dataset_name, dataset_name, dataset_name)
            return 1

    # Step 6 — Evaluation metrics: primary accuracy %, secondary accuracy per reasoning type (NextQA)
    model.eval()
    val_correct, val_total = 0, 0
    val_by_type = {}  # reasoning_type -> (correct, total)
    with torch.no_grad():
        for batch in val_dl:
            targets = batch["target"].to(device)
            if use_precomputed:
                B = targets.shape[0]
                emb = {k: batch[k].to(device).float() for k in batch if k not in ("target", "reasoning_type") and isinstance(batch.get(k), torch.Tensor)}
                if "pressure" not in emb:
                    emb["pressure"] = torch.zeros(B, PRESSURE_DIM, device=device, dtype=torch.float32)
                if "emg" not in emb:
                    emb["emg"] = torch.zeros(B, EMG_DIM, device=device, dtype=torch.float32)
                if "text" not in emb:
                    emb["text"] = torch.zeros(B, TEXT_DIM, device=device, dtype=torch.float32)
                # NextQA has no audio; zeros for audio when missing
                if "audio" not in emb:
                    emb["audio"] = torch.zeros(B, AUDIO_DIM, device=device, dtype=torch.float32)
                for k in emb:
                    emb[k] = torch.where(torch.isfinite(emb[k]), emb[k], torch.zeros_like(emb[k])).clamp(-30.0, 30.0)
            else:
                frames = batch["frame"].to(device).float()
                if args.vision_encoder == "viscop":
                    frames_in = frames.clamp(0.0, 1.0)
                else:
                    frames_in = (frames.clamp(0.0, 1.0) - clip_mean) / clip_std
                v_emb = vision(frames_in)
                if v_emb.dim() == 3:
                    v_emb = v_emb.mean(dim=1)
                B = v_emb.shape[0]
                emb = {"vision_camera1": v_emb}
                if load_from_sdata:
                    emb["vision_camera2"] = torch.zeros_like(v_emb, device=device)
                    emb["audio"] = torch.zeros(B, AUDIO_DIM, device=device, dtype=v_emb.dtype)
                    emb["text"] = torch.zeros(B, TEXT_DIM, device=device, dtype=v_emb.dtype)
                    emb["pressure"] = torch.zeros(B, PRESSURE_DIM, device=device, dtype=v_emb.dtype)
                    emb["emg"] = torch.zeros(B, EMG_DIM, device=device, dtype=v_emb.dtype)
                if has_text and "text" in batch and text_encoder_module is not None:
                    txt_emb = text_encoder_module(batch["text"])
                    if load_from_sdata and txt_emb.shape[-1] != TEXT_DIM:
                        pad = TEXT_DIM - txt_emb.shape[-1]
                        if pad > 0:
                            txt_emb = torch.cat([txt_emb, torch.zeros(txt_emb.shape[0], pad, device=device, dtype=txt_emb.dtype)], dim=-1)
                    emb["text"] = txt_emb
            logits, _, _ = model(emb, return_kl=True)
            pred = logits.argmax(1)
            val_correct += (pred == targets).sum().item()
            val_total += len(targets)
            if "reasoning_type" in batch and batch["reasoning_type"] is not None:
                for i, rt in enumerate(batch["reasoning_type"]):
                    if rt is None or (isinstance(rt, str) and not rt.strip()):
                        continue
                    rt = str(rt).strip().lower()
                    if rt not in val_by_type:
                        val_by_type[rt] = [0, 0]
                    val_by_type[rt][1] += 1
                    if pred[i].item() == targets[i].item():
                        val_by_type[rt][0] += 1
    val_acc = 100.0 * val_correct / val_total if val_total else 0.0
    log.info("--- Evaluation (validation) ---")
    log.info("Primary: accuracy = %.2f%%", val_acc)
    if val_by_type:
        log.info("Secondary (accuracy per reasoning type):")
        for rt in ("causal", "temporal", "descriptive"):
            if rt in val_by_type:
                c, t = val_by_type[rt]
                log.info("  %s = %.2f%% (%d / %d)", rt, 100.0 * c / t if t else 0.0, c, t)
        for rt, (c, t) in sorted(val_by_type.items()):
            if rt not in ("causal", "temporal", "descriptive"):
                log.info("  %s = %.2f%% (%d / %d)", rt, 100.0 * c / t if t else 0.0, c, t)

    ckpt = {
        "model_state": model.action_head.state_dict(),
        "fusion_state": model.fusion.state_dict(),
        "movement_state": model.movement_head.state_dict() if model.movement_head else None,
        "num_classes": n_class,
        "fusion_dim": fusion_dim,
        "dataset": dataset_name,
    }
    if vision is not None and (not args.freeze_vision or unfreeze_vis_n > 0):
        ckpt["vision_state"] = vision.state_dict()
    if text_encoder_module is not None and not args.freeze_text:
        ckpt["text_state"] = text_encoder_module.state_dict()

    # Save only to the canonical model file (single file, no extra checkpoints)
    save_path = model_file if model_file is not None else out_dir / "checkpoint.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, save_path)
    log.info("Saved: %s", save_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
