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
from torch.utils.data import Subset
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

from mmfuse.preprocessing.preprocessor import VisionPreprocessor, AudioPreprocessor
from mmfuse.encoders.vision_encoder import VisionEncoder
from mmfuse.encoders.audio_encoder_learnable import AudioEncoder as LearnableAudioEncoder
from mmfuse.encoders.audio_encoder import Wav2VecPooledEncoder
from mmfuse.encoders.audio_encoder_whisper import WhisperAudioEncoder
from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention


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
        data = torch.load(self.samples[idx], map_location='cpu', weights_only=True)
        return {
            'vision_camera1': data['vision_camera1'],
            'vision_camera2': data['vision_camera2'],
            'audio': data['audio'],
            'target': data['target'],
        }


def build_embedding_precomputed(batch, device, encoders):
    """Use precomputed embeddings; only fusion + classifier run."""
    v1 = torch.stack([s['vision_camera1'] for s in batch]).to(device).float()
    v2 = torch.stack([s['vision_camera2'] for s in batch]).to(device).float()
    a = torch.stack([s['audio'] for s in batch]).to(device).float()
    # Replace NaN/Inf with 0 to avoid training instability (e.g. from VisCoP)
    for t in (v1, v2, a):
        t[~torch.isfinite(t)] = 0.0
    pressures = torch.zeros(len(batch), 2, device=device)
    emgs = torch.zeros(len(batch), 4, device=device)

    p_emb = encoders['pressure'](pressures)
    e_emb = encoders['emg'](emgs)

    embeddings = {
        'vision_camera1': v1,
        'vision_camera2': v2,
        'audio': a,
        'pressure': p_emb,
        'emg': e_emb,
    }
    fused, kl_losses = encoders['fusion'](embeddings, return_kl=True)
    return fused, kl_losses


def build_embedding(batch, device, encoders):
    vis_imgs = [s['frame'] for s in batch]
    vis_imgs2 = [s['frame2'] for s in batch]
    auds = [s['audio'] for s in batch]
    # Placeholder: pressure (2), emg (4) features per sample to match encoder input dims
    pressures = [np.zeros((1, 2)) for _ in batch]
    emgs = [np.zeros((1, 4)) for _ in batch]

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

    embeddings = {
        'vision_camera1': v_emb1,
        'vision_camera2': v_emb2,
        'audio': a_emb,
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
        audio_dim = emb_config.get('audio_dim', 768)
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
        vision_dim = 512  # CLIP
        audio_dim = 768
        build_emb_fn = build_embedding

    # 80/20 stratified train/test split
    from sklearn.model_selection import train_test_split
    if use_precomputed:
        labels = [torch.load(p, map_location='cpu', weights_only=True)['target'] for p in ds.samples]
    else:
        labels = [s[3] for s in ds.samples]
    train_idx, test_idx = train_test_split(
        range(len(ds)), test_size=0.2, stratify=labels, random_state=42
    )
    train_ds = Subset(ds, train_idx)
    test_ds = Subset(ds, test_idx)
    num_workers = 4 if use_precomputed else 0
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    n_per_part = len(ds) // num_classes if num_classes else 0
    log.info("Dataset: %d samples (%d per part), %d classes", len(ds), n_per_part, num_classes)
    log.info("Train: %d | Test: %d (80/20 split)", len(train_idx), len(test_idx))

    log.info("Building model (vision_dim=%d, audio_dim=%d, fusion) ...", vision_dim, audio_dim)

    pressure = PressureSensorEncoder(output_dim=256, input_features=2).to(device)
    emg = EMGSensorEncoder(output_dim=256, num_channels=3, input_features=4).to(device)

    modality_dims = {
        'vision_camera1': vision_dim,
        'vision_camera2': vision_dim,
        'audio': audio_dim,
        'pressure': 256,
        'emg': 256,
    }
    fusion = MultimodalFusionWithAttention(
        modality_dims=modality_dims,
        fusion_dim=args.fusion_dim,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    if not use_precomputed:
        vision = VisionEncoder(device=str(device)).to(device)
        if args.audio_encoder == 'wav2vec':
            audio = Wav2VecPooledEncoder(frozen=True, device=str(device)).to(device)
        elif args.audio_encoder == 'whisper':
            audio = WhisperAudioEncoder(frozen=True, device=str(device)).to(device)
        else:
            audio = LearnableAudioEncoder(device=str(device)).to(device)
        if args.freeze_encoders:
            for m in [vision, audio, pressure, emg]:
                for p in m.parameters():
                    p.requires_grad = False
        encoders = {'vision': vision, 'audio': audio, 'pressure': pressure, 'emg': emg, 'fusion': fusion}
    else:
        encoders = {'pressure': pressure, 'emg': emg, 'fusion': fusion}

    model = ActionClassifier(embedding_dim=args.fusion_dim, num_classes=num_classes).to(device)
    trainable_params = list(model.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print("MODEL: MultimodalFusionWithAttention + ActionClassifier (sdata)")
    print("=" * 60)
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
    ]
    if not use_precomputed:
        components.insert(0, ("vision (CLIP)", vision))
        components.insert(1, (f"audio ({args.audio_encoder})", audio))
    print("\n--- Parameter counts")
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

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        fusion.train()
        epoch_loss = 0.0
        epoch_kl = 0.0
        correct = 0
        total = 0
        kl_losses = {}

        for batch_idx, batch in enumerate(train_dl):
            if batch_idx == 0:
                batch_start = time.perf_counter()
            targets = torch.tensor([s['target'] for s in batch], dtype=torch.long).to(device)

            fused, kl_losses = build_emb_fn(batch, device, encoders)
            logits = model(fused)

            loss_bc = criterion(logits, targets)
            loss_kl = sum(kl_losses.values()) if kl_losses else torch.tensor(0.0, device=device)
            loss = loss_bc + args.kl_weight * loss_kl

            optimizer.zero_grad()
            loss.backward()
            if torch.isfinite(loss).all():
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(fusion.parameters()), max_norm=1.0)
                optimizer.step()

            lb = loss_bc.item()
            epoch_loss += lb if (lb == lb and abs(lb) != float('inf')) else 0.0
            if kl_losses:
                for v in kl_losses.values():
                    kv = v.item()
                    epoch_kl += kv if (kv == kv and abs(kv) != float('inf')) else 0.0
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

        # Test evaluation
        model.eval()
        fusion.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for batch in test_dl:
                targets = torch.tensor([s['target'] for s in batch], dtype=torch.long).to(device)
                fused, _ = build_emb_fn(batch, device, encoders)
                logits = model(fused)
                pred = logits.argmax(dim=1)
                test_correct += (pred == targets).sum().item()
                test_total += len(batch)
        test_acc = test_correct / test_total if test_total > 0 else 0
        model.train()
        fusion.train()

        epoch_sec = time.perf_counter() - epoch_start
        epoch_times.append(epoch_sec)
        avg_epoch_sec = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.epochs - (epoch + 1)
        eta_sec = avg_epoch_sec * remaining_epochs if remaining_epochs > 0 else 0
        eta_str = f" | ETA {eta_sec/60:.1f}min" if remaining_epochs > 0 else ""

        log.info("Epoch %d/%d done in %.1fs | train_loss=%.4f train_acc=%.4f test_acc=%.4f%s%s",
                 epoch + 1, args.epochs, epoch_sec, epoch_loss / train_count, train_acc, test_acc, kl_str, eta_str)

        audio_enc = emb_config.get('audio_encoder', args.audio_encoder) if emb_config else args.audio_encoder
        vision_enc = emb_config.get('vision_encoder', 'clip') if emb_config else 'clip'
        ckpt = {
            'model_state': model.state_dict(),
            'fusion_state': fusion.state_dict(),
            'num_classes': num_classes,
            'epoch': epoch + 1,
            'audio_encoder': audio_enc,
            'vision_encoder': vision_enc,
            'vision_dim': vision_dim,
            'fusion_dim': args.fusion_dim,
        }
        ckpt_path = os.path.join(args.out_dir, f'ckpt_sdata_epoch_{epoch+1}.pt')
        torch.save(ckpt, ckpt_path)
        log.info("Checkpoint saved: %s", ckpt_path)

    total_train_sec = sum(epoch_times)
    log.info("=" * 60)
    log.info("TRAINING COMPLETE | total time %.1fs (%.1f min)", total_train_sec, total_train_sec / 60)
    log.info("=" * 60)


def main():
    p = argparse.ArgumentParser()
    _default_dataset = str(Path(__file__).resolve().parent.parent / 'dataset' / 'sdata')
    p.add_argument('--dataset', default=_default_dataset, help='Path to sdata folder')
    p.add_argument('--out-dir', default='checkpoints', help='Where to save checkpoints')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='Device (default: cuda if available)')
    p.add_argument('--freeze-encoders', action='store_true')
    p.add_argument('--kl-weight', type=float, default=0.1)
    p.add_argument('--num-heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--cross-pair', action='store_true', help='Pair each audio with all video pairs in same part (larger dataset)')
    p.add_argument('--augment-variations', type=int, default=16, help='Video augmentations per sample (default 16)')
    p.add_argument('--audio-encoder', choices=['learnable', 'wav2vec', 'whisper'], default='wav2vec',
                   help='Audio encoder: learnable CNN, wav2vec2-base (frozen), or whisper-small (frozen)')
    p.add_argument('--fusion-dim', type=int, default=256,
                   help='Fusion embedding dim (default 256; use 512 for original, smaller = fewer params)')
    p.add_argument('--use-precomputed', action='store_true',
                   help='Use precomputed embeddings (run scripts/precompute_sdata_embeddings.py first)')
    p.add_argument('--embeddings-dir', type=str, default=None,
                   help='Path to precomputed embeddings dir (required if --use-precomputed)')
    args = p.parse_args()

    if args.use_precomputed and not args.embeddings_dir:
        p.error('--embeddings-dir required when --use-precomputed')

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
