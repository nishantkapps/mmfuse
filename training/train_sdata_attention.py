#!/usr/bin/env python3
"""
Train MultimodalFusionWithAttention on sdata folder structure.
"""
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
    force=True
)
log = logging.getLogger(__name__)

# Dataset layout: dataset/sdata/part1/..partN/participant_id/*_c1_*.mp4, *_c2_*.mp4, *_m1_*.wav
import argparse
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import Subset
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from sklearn.metrics import precision_recall_fscore_support

from mmfuse.preprocessing.preprocessor import VisionPreprocessor, AudioPreprocessor
from mmfuse.encoders.vision_encoder import VisionEncoder
from mmfuse.encoders.vision_encoder_viscop import VisCoPVisionEncoder
from mmfuse.encoders.audio_encoder_learnable import AudioEncoder as LearnableAudioEncoder
from mmfuse.encoders.audio_encoder import Wav2VecPooledEncoder
from mmfuse.encoders.audio_encoder_whisper import WhisperAudioEncoder
from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention

try:
    from config_modality import FUSION_DIM, get_modality_dims, AUDIO_DIM, TEXT_DIM, PRESSURE_DIM, EMG_DIM
except ImportError:
    import sys
    _proj = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_proj))
    from config_modality import FUSION_DIM, get_modality_dims, AUDIO_DIM, TEXT_DIM, PRESSURE_DIM, EMG_DIM


def _unfreeze_last_n_layers(encoder_module, n: int, encoder_kind: str):
    """
    Unfreeze only the last n transformer layers of an encoder. Rest stays frozen.
    encoder_module: wrapper (e.g. VisCoPVisionEncoder, Wav2VecPooledEncoder) with .model.
    encoder_kind: 'viscop' | 'clip' | 'wav2vec' | 'whisper'
    Returns number of layers unfrozen (0 if structure not found).
    """
    if n <= 0:
        return 0
    # Ensure full encoder is frozen first
    for p in encoder_module.parameters():
        p.requires_grad = False
    root = getattr(encoder_module, 'model', encoder_module)
    layers = None
    if encoder_kind == 'wav2vec':
        # Wav2Vec2Model: .encoder.layers
        if hasattr(root, 'encoder') and hasattr(root.encoder, 'layers'):
            layers = root.encoder.layers
    elif encoder_kind == 'whisper':
        if hasattr(root, 'encoder') and hasattr(root.encoder, 'layers'):
            layers = root.encoder.layers
    elif encoder_kind == 'viscop':
        # VisCoP / Qwen2-style: .model.model.layers, .model.layers, or .layers
        if hasattr(root, 'model') and hasattr(root.model, 'model') and hasattr(root.model.model, 'layers'):
            layers = root.model.model.layers
        elif hasattr(root, 'model') and hasattr(root.model, 'layers'):
            layers = root.model.layers
        elif hasattr(root, 'layers'):
            layers = root.layers
    elif encoder_kind == 'clip':
        # CLIP: .visual.transformer.resblocks or .transformer.resblocks
        if hasattr(root, 'visual') and hasattr(root.visual, 'transformer'):
            blocks = getattr(root.visual.transformer, 'resblocks', None) or getattr(root.visual.transformer, 'layers', None)
            if blocks is not None:
                layers = blocks
        elif hasattr(root, 'transformer'):
            blocks = getattr(root.transformer, 'resblocks', None) or getattr(root.transformer, 'layers', None)
            if blocks is not None:
                layers = blocks
    if layers is None or not isinstance(layers, nn.ModuleList):
        log.warning("Unfreeze last %d layers: no layer list found for %s (encoder may stay fully frozen)", n, encoder_kind)
        return 0
    total = len(layers)
    take = min(n, total)
    for i in range(total - take, total):
        for p in layers[i].parameters():
            p.requires_grad = True
    log.info("Unfroze last %d layer(s) of %s (total %d)", take, encoder_kind, total)
    return take


def _get_audio_path(pdir: Path):
    """sdata uses {participant}_m1_part{N}.wav (e.g. p012_m1_part4.wav) per participant folder."""
    m1 = list(pdir.glob('*_m1_*.wav'))
    return m1[0] if m1 else None


def _augment_frame(frame: np.ndarray, variation_id: int) -> np.ndarray:
    """Apply deterministic augmentation based on variation_id (0..N-1). Returns RGB numpy uint8."""
    if frame is None or frame.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    out = frame.copy().astype(np.float32)
    # 16 variations: flip (2) × brightness (4) × contrast (2)
    if (variation_id // 8) % 2:
        out = np.ascontiguousarray(out[:, ::-1, :])
    brightness = [0.85, 0.95, 1.05, 1.15][(variation_id // 2) % 4]
    contrast = [0.9, 1.1][variation_id % 2]
    out = out * brightness
    out = (out - 127.5) * contrast + 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


class SDataDataset(Dataset):
    """
    Loads from sdata/partX/participant_id/ with *_c1_*.mp4 and *_c2_*.mp4.
    Label = part index (action class).
    cross_pair_audio_video: if True, each audio is paired with ALL video pairs in the same part.
    augment_variations: number of augmented versions per sample (default 16).
    """
    def __init__(self, root_dir, device='cpu', cross_pair_audio_video: bool = False, augment_variations: int = 16):
        self.root_dir = Path(root_dir)
        self.device = device
        self.cross_pair = cross_pair_audio_video
        self.augment_variations = max(1, augment_variations)
        self.vprep = VisionPreprocessor(image_size=(224, 224))
        self.aprep = AudioPreprocessor(sample_rate=16000, duration=2.5)

        self.samples = []
        part_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('part')])
        self.part_to_label = {p.name: i for i, p in enumerate(part_dirs)}
        self.num_classes = len(part_dirs)

        for part_dir in part_dirs:
            label = self.part_to_label[part_dir.name]
            video_pairs = []
            audio_paths = []
            for pdir in part_dir.iterdir():
                if not pdir.is_dir():
                    continue
                cam1 = list(pdir.glob('*_c1_*.mp4'))
                cam2 = list(pdir.glob('*_c2_*.mp4'))
                if not (cam1 and cam2):
                    continue
                video_pairs.append((cam1[0], cam2[0]))
                audio_paths.append(_get_audio_path(pdir))

            # 1) Generate all augmented video pairs first: each (cam1,cam2) -> 16 variants
            augmented_video_pairs = []
            for cam1, cam2 in video_pairs:
                for v in range(self.augment_variations):
                    augmented_video_pairs.append((cam1, cam2, v))

            # 2) Cross-pair: each audio with all augmented video pairs
            if cross_pair_audio_video:
                for audio_path in audio_paths:
                    for cam1, cam2, v in augmented_video_pairs:
                        self.samples.append((audio_path, cam1, cam2, label, v))
            else:
                for i, (cam1, cam2, v) in enumerate(augmented_video_pairs):
                    audio_path = audio_paths[i // self.augment_variations] if i // self.augment_variations < len(audio_paths) else None
                    self.samples.append((audio_path, cam1, cam2, label, v))

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, path):
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        mid = max(0, n // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def __getitem__(self, idx):
        audio_path, cam1_path, cam2_path, label, variation_id = self.samples[idx]
        frame1 = self._load_frame(cam1_path)
        frame2 = self._load_frame(cam2_path)
        frame1 = _augment_frame(frame1, variation_id)
        frame2 = _augment_frame(frame2, variation_id)
        audio_str = str(audio_path) if audio_path is not None else None

        return {
            'frame': frame1,
            'frame2': frame2,
            'audio': audio_str,
            'pressure': None,
            'emg': None,
            'target': label,
        }


def collate_fn(batch):
    return batch


class PrecomputedSDataDataset(Dataset):
    """Loads precomputed embeddings from disk. No video/audio I/O during training."""
    def __init__(self, embeddings_dir: Path, config: dict):
        self.embeddings_dir = Path(embeddings_dir)
        self.num_classes = config.get('num_classes', 8)
        self.samples = list(sorted(self.embeddings_dir.glob('*.pt')))
        if not self.samples:
            raise RuntimeError(f"No .pt files in {embeddings_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx], weights_only=True)
        # SData has no direct text; use zeros when "text" key missing (backward compat)
        text_emb = data.get('text', torch.zeros(TEXT_DIM))
        return {
            'vision_camera1': data['vision_camera1'],
            'vision_camera2': data['vision_camera2'],
            'audio': data['audio'],
            'text': text_emb,
            'target': data['target'],
        }


def build_embedding_precomputed(batch, device, encoders):
    """Use precomputed embeddings; only fusion + classifier run."""
    v1 = torch.stack([s['vision_camera1'] for s in batch]).to(device).float()
    v2 = torch.stack([s['vision_camera2'] for s in batch]).to(device).float()
    a = torch.stack([s['audio'] for s in batch]).to(device).float()
    txt = torch.stack([s['text'] for s in batch]).to(device).float()
    # Replace NaN/Inf with 0 to avoid training instability (e.g. from VisCoP)
    for t in (v1, v2, a, txt):
        t[~torch.isfinite(t)] = 0.0
    pressures = torch.zeros(len(batch), 2, device=device)
    emgs = torch.zeros(len(batch), 4, device=device)

    p_emb = encoders['pressure'](pressures)
    e_emb = encoders['emg'](emgs)

    embeddings = {
        'vision_camera1': v1,
        'vision_camera2': v2,
        'audio': a,
        'text': txt,
        'pressure': p_emb,
        'emg': e_emb,
    }
    fused, kl_losses = encoders['fusion'](embeddings, return_kl=True)
    return fused, kl_losses


def build_embedding(batch, device, encoders, vision_encoder='clip'):
    vis_imgs = [s['frame'] for s in batch]
    vis_imgs2 = [s['frame2'] for s in batch]
    auds = [s['audio'] for s in batch]
    # Placeholder: pressure (2), emg (4) features per sample to match encoder input dims
    pressures = [np.zeros((1, 2)) for _ in batch]
    emgs = [np.zeros((1, 4)) for _ in batch]

    if vision_encoder == 'viscop':
        # VisCoP expects (B, 3, H, W) in [0, 1]; resize to 224x224, no ImageNet norm
        def to_viscop(imgs):
            out = []
            for img in imgs:
                if isinstance(img, np.ndarray):
                    img = cv2.resize(img, (224, 224))
                    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                else:
                    t = img
                out.append(t)
            return torch.stack(out).to(device)
        vis_batch = to_viscop(vis_imgs)
        vis2_batch = to_viscop(vis_imgs2)
    else:
        vprep = VisionPreprocessor()
        vis_tensors = [vprep.preprocess(img) for img in vis_imgs]
        vis_batch = torch.stack(vis_tensors).to(device)
        vis2_tensors = [vprep.preprocess(img) for img in vis_imgs2]
        vis2_batch = torch.stack(vis2_tensors).to(device)

    aprep = AudioPreprocessor()
    target_samples = int(aprep.duration * aprep.sample_rate)
    audio_tensors = []
    for a in auds:
        if a is None or not Path(a).exists():
            audio_tensors.append(torch.zeros(target_samples))
        else:
            try:
                audio_tensors.append(aprep.preprocess(a))
            except Exception:
                audio_tensors.append(torch.zeros(target_samples))
    audio_batch = torch.stack(audio_tensors).to(device)

    pressure_feats = [torch.tensor(p.mean(axis=0) if p.ndim > 1 else p.mean()) for p in pressures]
    emg_feats = [torch.tensor(e.mean(axis=0) if e.ndim > 1 else e.mean()) for e in emgs]
    pressure_batch = torch.stack([f.float() for f in pressure_feats]).to(device)
    emg_batch = torch.stack([f.float() for f in emg_feats]).to(device)

    with torch.no_grad():
        v_emb1 = encoders['vision'](vis_batch)
        v_emb2 = encoders['vision'](vis2_batch)
        a_emb = encoders['audio'](audio_batch)
        p_emb = encoders['pressure'](pressure_batch.unsqueeze(0)) if pressure_batch.dim() == 1 else encoders['pressure'](pressure_batch)
        e_emb = encoders['emg'](emg_batch.unsqueeze(0)) if emg_batch.dim() == 1 else encoders['emg'](emg_batch)

    # SData has no direct text input; pass zeros for text modality
    text_zeros = torch.zeros(len(batch), TEXT_DIM, device=device)
    embeddings = {
        'vision_camera1': v_emb1,
        'vision_camera2': v_emb2,
        'audio': a_emb,
        'text': text_zeros,
        'pressure': p_emb,
        'emg': e_emb,
    }
    fused, kl_losses = encoders['fusion'](embeddings, return_kl=True)
    return fused, kl_losses


class ActionClassifier(nn.Module):
    """Classification head: fused embedding -> num_classes."""
    def __init__(self, embedding_dim=512, num_classes=8):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MovementHead(nn.Module):
    """Regression head: fused embedding -> (delta_along, delta_lateral, magnitude)."""
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 3)

    def forward(self, x):
        return self.fc(x)


def load_movement_targets(config_path: str, num_classes: int) -> torch.Tensor:
    """Load (delta_along, delta_lateral, magnitude) per action_id from YAML. Returns (num_classes, 3)."""
    path = Path(config_path)
    if not path.is_absolute():
        proj_root = Path(__file__).resolve().parent.parent
        path = proj_root / path
    if not path.exists():
        log.warning("Movement config not found at %s; using zeros", config_path)
        return torch.zeros(num_classes, 3)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    movements = cfg.get('movements', [])
    targets = []
    for i in range(num_classes):
        if i < len(movements):
            m = movements[i]
            targets.append([m.get('delta_along', 0), m.get('delta_lateral', 0), m.get('magnitude', 0)])
        else:
            targets.append([0, 0, 0])
    return torch.tensor(targets, dtype=torch.float32)


def train(args):
    device = torch.device(args.device)
    log.info("=" * 60)
    log.info("TRAINING STARTED | device=%s | cuda_available=%s", device, torch.cuda.is_available())
    if device.type == 'cuda':
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    log.info("=" * 60)

    use_precomputed = getattr(args, 'use_precomputed', False)
    embeddings_dir = getattr(args, 'embeddings_dir', None)

    emb_config = {}
    if use_precomputed and embeddings_dir:
        import json
        emb_dir = Path(embeddings_dir)
        with open(emb_dir / 'config.json') as f:
            emb_config = json.load(f)
        log.info("Loading precomputed embeddings from %s ...", emb_dir)
        ds = PrecomputedSDataDataset(emb_dir, emb_config)
        num_classes = emb_config.get('num_classes', 8)
        vision_dim = emb_config['vision_dim']
        audio_dim = emb_config.get('audio_dim', AUDIO_DIM)
        build_emb_fn = build_embedding_precomputed
    else:
        log.info("Loading dataset from %s ...", args.dataset)
        ds = SDataDataset(
            args.dataset,
            device=str(device),
            cross_pair_audio_video=args.cross_pair,
            augment_variations=args.augment_variations,
        )
        if len(ds) == 0:
            raise RuntimeError(f"No samples found in {args.dataset}. Expected partX/participant_id/*_c1_*.mp4, *_c2_*.mp4")
        num_classes = ds.num_classes
        vision_encoder = getattr(args, 'vision_encoder', 'clip')
        modality_dims_raw = get_modality_dims(vision_encoder)
        vision_dim = modality_dims_raw['vision_camera1']
        audio_dim = modality_dims_raw['audio']
        build_emb_fn = lambda b, d, e: build_embedding(b, d, e, vision_encoder=getattr(args, 'vision_encoder', 'clip'))

    # Train/test split: before augmentation (precomputed: use train_count/test_count; raw: split by pairs)
    from sklearn.model_selection import train_test_split
    if use_precomputed:
        train_count = emb_config.get('train_count')
        test_count = emb_config.get('test_count')
        n_samples = len(ds.samples)
        if train_count is not None and test_count is not None and train_count + test_count == n_samples:
            train_idx = list(range(0, train_count))
            test_idx = list(range(train_count, train_count + test_count))
            log.info("Precomputed: using split-before-augmentation (train=%d test=%d)", train_count, test_count)
        else:
            if train_count is not None or test_count is not None:
                log.warning("Precomputed: config train_count=%s test_count=%s but len(samples)=%d (mismatch). Using 90/10 split. Re-run precompute to get split-before-aug.",
                            train_count, test_count, n_samples)
            else:
                log.warning("Precomputed: no train_count/test_count in config; using 90/10 split. Re-run precompute with current script for split-before-aug.")
            labels = [torch.load(p, weights_only=True)['target'] for p in ds.samples]
            train_idx, test_idx = train_test_split(
                range(len(ds)), test_size=0.1, stratify=labels, random_state=42
            )
    else:
        # Split by base (cam1, cam2) first; train gets all augmentations, test gets v=0 only
        pair_to_label = {}
        for audio_path, cam1, cam2, label, v in ds.samples:
            key = (str(cam1), str(cam2))
            pair_to_label[key] = label
        unique_pairs = list(pair_to_label.keys())
        pair_labels = [pair_to_label[p] for p in unique_pairs]
        train_pairs, test_pairs = train_test_split(
            unique_pairs, test_size=0.1, stratify=pair_labels, random_state=42
        )
        train_pairs_set = set(train_pairs)
        test_pairs_set = set(test_pairs)
        train_idx = []
        test_idx = []
        for i, (audio_path, cam1, cam2, label, v) in enumerate(ds.samples):
            key = (str(cam1), str(cam2))
            if key in train_pairs_set:
                train_idx.append(i)
            elif key in test_pairs_set and v == 0:
                test_idx.append(i)
        log.info("Split before augmentation: %d train pairs (all aug) | %d test pairs (v=0 only)",
                 len(train_pairs), len(test_pairs))
    train_ds = Subset(ds, train_idx)
    test_ds = Subset(ds, test_idx)
    num_workers = 4 if use_precomputed else 0
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    n_per_part = len(ds) // num_classes if num_classes else 0
    log.info("Dataset: %d samples (%d per part), %d classes", len(ds), n_per_part, num_classes)
    log.info("Train: %d | Test: %d (90/10 split)", len(train_idx), len(test_idx))

    if use_precomputed:
        log.info("Building model (vision_dim=%d, audio_dim=%d from precomputed, fusion) ...", vision_dim, audio_dim)
    else:
        log.info("Building model (vision_dim=%d, audio_dim=%d, fusion) ...", vision_dim, audio_dim)
        if getattr(args, 'finetune_encoders', False) and not args.freeze_encoders:
            uv = getattr(args, 'unfreeze_vision_layers', 0)
            ua = getattr(args, 'unfreeze_audio_layers', 0)
            if uv > 0 or ua > 0:
                log.info("--finetune-encoders: unfreezing last %d vision / %d audio layer(s) only", uv or 0, ua or 0)
            else:
                log.info("--finetune-encoders: vision and audio encoders will be trained (all layers unfrozen)")

    movement_targets = load_movement_targets(args.movement_config, num_classes).to(device)

    pressure = PressureSensorEncoder(output_dim=PRESSURE_DIM, input_features=2).to(device)
    emg = EMGSensorEncoder(output_dim=EMG_DIM, num_channels=3, input_features=4).to(device)

    modality_dims = get_modality_dims(getattr(args, 'vision_encoder', 'clip'))
    modality_dims = {**modality_dims, 'vision_camera1': vision_dim, 'vision_camera2': vision_dim, 'audio': audio_dim}
    fusion_dim = getattr(args, 'fusion_dim', FUSION_DIM)
    fusion = MultimodalFusionWithAttention(
        modality_dims=modality_dims,
        fusion_dim=fusion_dim,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    if not use_precomputed:
        # When --finetune-encoders (and not --freeze-encoders): unfreeze vision/audio for finetuning.
        # If --unfreeze-vision-layers N or --unfreeze-audio-layers N (N>0), only last N layers are unfrozen to limit params.
        finetune_enc = getattr(args, 'finetune_encoders', False) and not args.freeze_encoders
        unfreeze_vis_n = max(0, getattr(args, 'unfreeze_vision_layers', 0))
        unfreeze_aud_n = max(0, getattr(args, 'unfreeze_audio_layers', 0))
        # Create vision frozen if: not finetuning, or finetuning with "last N layers" (we unfreeze after)
        freeze_vision = not finetune_enc or (unfreeze_vis_n > 0)
        # Audio: same; for learnable encoder "last N layers" has no effect so we unfreeze all when finetuning
        freeze_audio = not finetune_enc or (unfreeze_aud_n > 0 and args.audio_encoder in ('wav2vec', 'whisper'))
        if getattr(args, 'vision_encoder', 'clip') == 'viscop':
            vision = VisCoPVisionEncoder(
                model_path="viscop_trained_models/viscop_qwen2.5_7b_viscop-lora_egocentric-expert",
                device=str(device),
                frozen=freeze_vision,
            ).to(device)
        else:
            vision = VisionEncoder(device=str(device)).to(device)
            if freeze_vision:
                for p in vision.parameters():
                    p.requires_grad = False
        if args.audio_encoder == 'wav2vec':
            audio = Wav2VecPooledEncoder(frozen=freeze_audio, device=str(device)).to(device)
        elif args.audio_encoder == 'whisper':
            audio = WhisperAudioEncoder(frozen=freeze_audio, device=str(device)).to(device)
        else:
            audio = LearnableAudioEncoder(device=str(device)).to(device)
            if freeze_audio:
                for p in audio.parameters():
                    p.requires_grad = False
        if args.freeze_encoders:
            for m in [vision, audio, pressure, emg]:
                for p in m.parameters():
                    p.requires_grad = False
        # Unfreeze only last N layers when requested (vision/audio created frozen above)
        if finetune_enc and unfreeze_vis_n > 0:
            _unfreeze_last_n_layers(vision, unfreeze_vis_n, 'viscop' if getattr(args, 'vision_encoder', 'clip') == 'viscop' else 'clip')
        if finetune_enc and unfreeze_aud_n > 0 and args.audio_encoder in ('wav2vec', 'whisper'):
            _unfreeze_last_n_layers(audio, unfreeze_aud_n, args.audio_encoder)
        encoders = {'vision': vision, 'audio': audio, 'pressure': pressure, 'emg': emg, 'fusion': fusion}
    else:
        encoders = {'pressure': pressure, 'emg': emg, 'fusion': fusion}

    model = ActionClassifier(embedding_dim=fusion_dim, num_classes=num_classes).to(device)
    movement_head = MovementHead(embedding_dim=fusion_dim).to(device)
    trainable_params = list(model.parameters()) + list(fusion.parameters()) + list(movement_head.parameters())
    if not use_precomputed and getattr(args, 'finetune_encoders', False) and not args.freeze_encoders:
        trainable_params += list(vision.parameters()) + list(audio.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    modality_names = list(modality_dims.keys())
    print("\n" + "=" * 60)
    print("MODEL: MultimodalFusionWithAttention + ActionClassifier (sdata)")
    print("=" * 60)
    print("\n--- Modalities in fusion: {}".format(", ".join(modality_names)))
    if use_precomputed:
        print("     (vision_camera1, vision_camera2, audio, text from precomputed embeddings; no encoder modules)")
    print("\n--- Fusion pipeline (forward flow)")
    print("  1. Projections:  modality_emb -> Linear+BN+ReLU -> fusion_dim (per modality)")
    print("  2. KL Divergence (knowledge distillation):")
    print("       - vision_camera1 <-> vision_camera2  (camera alignment)")
    print("  3. Cross-Modal Attention: MultiheadAttention (num_heads={})".format(args.num_heads))
    print("  4. Fusion MLP -> fused embedding")
    print("\n--- Classifier")
    print(model)

    def _count(m):
        return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)

    components = [
        ("pressure", pressure),
        ("emg", emg),
        ("fusion (attention + KL)", fusion),
        ("classifier", model),
        ("movement_head", movement_head),
    ]
    if not use_precomputed:
        components.insert(0, (f"vision ({getattr(args, 'vision_encoder', 'clip')})", vision))
        components.insert(1, (f"audio ({args.audio_encoder})", audio))
    print("\n--- Parameter counts")
    if use_precomputed:
        print("  vision_camera1, vision_camera2, audio, text: from precomputed embeddings (no encoder parameters)")
    total_all, trainable_all = 0, 0
    for name, m in components:
        t, tr = _count(m)
        total_all += t
        trainable_all += tr
        status = "trainable" if tr > 0 else "frozen"
        print(f"  {name}: {t:,} ({status})")
    log.info("TOTAL: %s | Trainable: %s", f"{total_all:,}", f"{trainable_all:,}")
    log.info("=" * 60)

    total_batches = len(train_dl)
    log_interval = max(1, total_batches // 10)  # log ~10 times per epoch
    epoch_times = []
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1': [],
        'epoch_sec': [],
    }

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        fusion.train()
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_contrast = 0.0
        epoch_movement = 0.0
        correct = 0
        total = 0
        kl_losses = {}

        for batch_idx, batch in enumerate(train_dl):
            if batch_idx == 0:
                batch_start = time.perf_counter()
            targets = torch.tensor([s['target'] for s in batch], dtype=torch.long).to(device)

            fused, kl_losses = build_emb_fn(batch, device, encoders)
            logits = model(fused)
            movement_pred = movement_head(fused)

            loss_bc = criterion(logits, targets)
            #loss_kl = sum(kl_losses.values()) if kl_losses else torch.tensor(0.0, device=device)
            movement_targets_batch = movement_targets[targets]
            loss_movement = nn.functional.mse_loss(movement_pred, movement_targets_batch)
            #loss = loss_bc + args.kl_weight * loss_kl + args.movement_weight * loss_movement

            kl_loss = kl_losses.get("kl_camera", torch.tensor(0.0, device=device))
            contrast_loss = kl_losses.get("contrastive_3way", torch.tensor(0.0, device=device))
            loss = (
                loss_bc
                + args.kl_weight * kl_loss
                + args.contrast_weight * contrast_loss
                + args.movement_weight * loss_movement
            )

            optimizer.zero_grad()
            loss.backward()
            if torch.isfinite(loss).all():
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(fusion.parameters()) + list(movement_head.parameters()),
                    max_norm=1.0)
                optimizer.step()

            lb = loss_bc.item()
            epoch_loss += lb if (lb == lb and abs(lb) != float('inf')) else 0.0
            if args.movement_weight > 0:
                epoch_movement += loss_movement.item() if torch.isfinite(loss_movement) else 0.0
            #if kl_losses:
            #    for v in kl_losses.values():
            #        kv = v.item()
            #        epoch_kl += kv if (kv == kv and abs(kv) != float('inf')) else 0.0
            if kl_losses:
                kl_val = kl_losses.get("kl_camera")
                contrast_val = kl_losses.get("contrastive_3way")
                if kl_val is not None:
                    kv = kl_val.item()
                    epoch_kl += kv if (kv == kv and abs(kv) != float('inf')) else 0.0
                if contrast_val is not None:
                    cv = contrast_val.item()
                    epoch_contrast += cv if (cv == cv and abs(cv) != float('inf')) else 0.0        
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += len(batch)

            if batch_idx == 0:
                first_batch_sec = time.perf_counter() - batch_start
                log.info("Epoch %d/%d: first batch done in %.1fs (training in progress on %s)", epoch + 1, args.epochs, first_batch_sec, device)
            elif (batch_idx + 1) % log_interval == 0:
                pct = 100.0 * (batch_idx + 1) / total_batches
                log.info("Epoch %d/%d: batch %d/%d (%.0f%%)", epoch + 1, args.epochs, batch_idx + 1, total_batches, pct)

        train_count = len(train_dl)
        train_acc = correct / total if total > 0 else 0
        kl_str = f" | KL={epoch_kl/train_count:.4f}" if kl_losses else ""
        contrast_str = f" | Contrast={epoch_contrast/train_count:.4f}" if kl_losses else ""

        # Test evaluation
        model.eval()
        fusion.eval()
        movement_head.eval()
        test_correct, test_total = 0, 0
        test_movement_mse = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_dl:
                targets = torch.tensor([s['target'] for s in batch], dtype=torch.long).to(device)
                fused, _ = build_emb_fn(batch, device, encoders)
                logits = model(fused)
                movement_pred = movement_head(fused)
                pred = logits.argmax(dim=1)
                test_correct += (pred == targets).sum().item()
                test_total += len(batch)
                y_true.extend(targets.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())
                mt = movement_targets[targets]
                test_movement_mse += nn.functional.mse_loss(movement_pred, mt).item() * len(batch)
        test_acc = test_correct / test_total if test_total > 0 else 0
        test_movement_mse = test_movement_mse / test_total if test_total > 0 else 0.0

        # Macro precision/recall/F1 over test set
        if y_true and y_pred:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average='macro',
                zero_division=0,
            )
        else:
            prec, rec, f1 = 0.0, 0.0, 0.0

        model.train()
        fusion.train()
        movement_head.train()

        epoch_sec = time.perf_counter() - epoch_start
        epoch_times.append(epoch_sec)
        avg_epoch_sec = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.epochs - (epoch + 1)
        eta_sec = avg_epoch_sec * remaining_epochs if remaining_epochs > 0 else 0
        eta_str = f" | ETA {eta_sec/60:.1f}min" if remaining_epochs > 0 else ""

        movement_str = f" | mov={epoch_movement/train_count:.4f} test_mov={test_movement_mse:.4f}" if args.movement_weight > 0 else ""
        prf_str = f" | prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}"
        log.info(
            "Epoch %d/%d done in %.1fs | train_loss=%.4f train_acc=%.4f test_acc=%.4f%s%s%s%s%s",
            epoch + 1,
            args.epochs,
            epoch_sec,
            epoch_loss / train_count,
            train_acc,
            test_acc,
            kl_str,
            contrast_str,
            movement_str,
            prf_str,
            eta_str,
        )

        history['train_loss'].append(epoch_loss / train_count)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['test_precision'].append(prec)
        history['test_recall'].append(rec)
        history['test_f1'].append(f1)
        history['epoch_sec'].append(epoch_sec)

        audio_enc = emb_config.get('audio_encoder', args.audio_encoder) if emb_config else args.audio_encoder
        vision_enc = emb_config.get('vision_encoder', getattr(args, 'vision_encoder', 'clip')) if emb_config else getattr(args, 'vision_encoder', 'clip')
        ckpt = {
            'model_state': model.state_dict(),
            'fusion_state': fusion.state_dict(),
            'movement_state': movement_head.state_dict(),
            'movement_targets': movement_targets.cpu(),
            'num_classes': num_classes,
            'epoch': epoch + 1,
            'audio_encoder': audio_enc,
            'vision_encoder': vision_enc,
            'vision_dim': vision_dim,
            'fusion_dim': fusion_dim,
        }
        ckpt_path = os.path.join(args.out_dir, f'ckpt_sdata_epoch_{epoch+1}.pt')
        torch.save(ckpt, ckpt_path)
        log.info("Checkpoint saved: %s", ckpt_path)

    # Update canonical model file once at end so finetuning can use the same path every time
    if getattr(args, 'model_file', None):
        model_path = args.model_file
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        final_ckpt = {
            'model_state': model.state_dict(),
            'fusion_state': fusion.state_dict(),
            'movement_state': movement_head.state_dict(),
            'movement_targets': movement_targets.cpu(),
            'num_classes': num_classes,
            'epoch': args.epochs,
            'audio_encoder': audio_enc,
            'vision_encoder': vision_enc,
            'vision_dim': vision_dim,
            'fusion_dim': fusion_dim,
        }
        torch.save(final_ckpt, model_path)
        log.info("Model file updated: %s", model_path)

    total_train_sec = sum(epoch_times)
    history_path = os.path.join(args.out_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        import json
        json.dump(history, f, indent=2)
    log.info("Training history saved: %s", history_path)
    log.info("=" * 60)
    log.info("TRAINING COMPLETE | total time %.1fs (%.1f min)", total_train_sec, total_train_sec / 60)
    log.info("=" * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='dataset/sdata', help='Path to sdata folder (relative to cwd)')
    p.add_argument('--out-dir', default='checkpoints', help='Where to save checkpoints (relative to cwd)')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='Device (default: cuda if available)')
    p.add_argument('--freeze-encoders', action='store_true',
                   help='Freeze all encoders (vision, audio, pressure, emg). Ignored if --use-precomputed.')
    p.add_argument('--finetune-encoders', action='store_true',
                   help='Train vision and audio encoders (unfreeze). Use for finetuning on other datasets. Ignored if --use-precomputed or --freeze-encoders.')
    p.add_argument('--unfreeze-vision-layers', type=int, default=0, metavar='N',
                   help='With --finetune-encoders: unfreeze only last N vision encoder layers (0 = unfreeze all). E.g. 1 or 2 to limit trainable params.')
    p.add_argument('--unfreeze-audio-layers', type=int, default=0, metavar='N',
                   help='With --finetune-encoders: unfreeze only last N audio encoder layers (0 = unfreeze all). E.g. 1 or 2 to limit trainable params.')
    p.add_argument('--kl-weight', type=float, default=0.05,help='Weight for KL loss')
    p.add_argument('--contrast-weight', type=float, default=0.1,help='Weight for contrastive alignment loss')
    p.add_argument('--num-heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--cross-pair', action='store_true', help='Pair each audio with all video pairs in same part (larger dataset)')
    p.add_argument('--augment-variations', type=int, default=16, help='Video augmentations per sample (default 16)')
    p.add_argument('--vision-encoder', choices=['clip', 'viscop'], default='clip',
                   help='Vision encoder: CLIP (512-dim) or VisCoP (3584-dim). For precomputed, use embeddings from precompute_sdata_embeddings.py --vision-encoder viscop')
    p.add_argument('--audio-encoder', choices=['learnable', 'wav2vec', 'whisper'], default='wav2vec',
                   help='Audio encoder: learnable CNN, wav2vec2-base, or whisper-small. Use --finetune-encoders to train (else frozen).')
    p.add_argument('--fusion-dim', type=int, default=FUSION_DIM,
                   help='Fusion embedding dim (default from config_modality.py, use 512 for finetune compatibility)')
    p.add_argument('--use-precomputed', action='store_true',
                   help='Use precomputed embeddings (run scripts/precompute_sdata_embeddings.py first)')
    p.add_argument('--embeddings-dir', type=str, default=None,
                   help='Path to precomputed embeddings dir (required if --use-precomputed)')
    p.add_argument('--movement-config', type=str, default='config/sdata_movement_config.yaml',
                   help='YAML mapping action_id -> (delta_along, delta_lateral, magnitude)')
    p.add_argument('--movement-weight', type=float, default=0.5,
                   help='Weight for movement loss (L = L_action + weight * L_movement)')
    p.add_argument('--model-file', type=str, default=None,
                   help='Path to canonical model file. After training, save final checkpoint here (e.g. checkpoints/model.pt) so finetuning can always use this path.')
    args = p.parse_args()

    if args.use_precomputed and not args.embeddings_dir:
        p.error('--embeddings-dir required when --use-precomputed')

    # Resolve relative paths from cwd (where you run the command)
    cwd = Path.cwd()
    args.dataset = str(cwd / args.dataset) if not os.path.isabs(args.dataset) else args.dataset
    args.out_dir = str(cwd / args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    if getattr(args, 'embeddings_dir', None):
        args.embeddings_dir = str(cwd / args.embeddings_dir) if not os.path.isabs(args.embeddings_dir) else args.embeddings_dir
    if getattr(args, 'model_file', None):
        args.model_file = str(cwd / args.model_file) if not os.path.isabs(args.model_file) else args.model_file
    args.movement_config = str(cwd / args.movement_config) if not os.path.isabs(args.movement_config) else args.movement_config

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
