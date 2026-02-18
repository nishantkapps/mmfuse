#!/usr/bin/env python3
"""
Precompute vision + audio embeddings for sdata dataset.
Run once, then train with --use-precomputed --embeddings-dir <path> for fast training.

Usage:
  python scripts/precompute_sdata_embeddings.py --dataset dataset/sdata --out-dir embeddings/sdata \\
      --vision-encoder viscop --audio-encoder wav2vec --cross-pair
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cv2

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def _get_audio_path(pdir: Path):
    m1 = list(pdir.glob('*_m1_*.wav'))
    return m1[0] if m1 else None


def _augment_frame(frame: np.ndarray, variation_id: int) -> np.ndarray:
    if frame is None or frame.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    out = frame.copy().astype(np.float32)
    if (variation_id // 8) % 2:
        out = np.ascontiguousarray(out[:, ::-1, :])
    brightness = [0.85, 0.95, 1.05, 1.15][(variation_id // 2) % 4]
    contrast = [0.9, 1.1][variation_id % 2]
    out = out * brightness
    out = (out - 127.5) * contrast + 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


def build_sample_list(root_dir: Path, cross_pair: bool, augment_variations: int):
    """Same logic as SDataDataset.samples."""
    samples = []
    part_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith('part')])
    part_to_label = {p.name: i for i, p in enumerate(part_dirs)}

    for part_dir in part_dirs:
        label = part_to_label[part_dir.name]
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

        augmented_video_pairs = []
        for cam1, cam2 in video_pairs:
            for v in range(augment_variations):
                augmented_video_pairs.append((cam1, cam2, v))

        if cross_pair:
            for audio_path in audio_paths:
                for cam1, cam2, v in augmented_video_pairs:
                    samples.append((audio_path, cam1, cam2, label, v))
        else:
            for i, (cam1, cam2, v) in enumerate(augmented_video_pairs):
                audio_path = audio_paths[i // augment_variations] if i // augment_variations < len(audio_paths) else None
                samples.append((audio_path, cam1, cam2, label, v))
    return samples


def load_frame(path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    mid = max(0, n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.zeros((224, 224, 3), dtype=np.uint8)


def main():
    _script_dir = Path(__file__).resolve().parent
    _project_root = _script_dir.parent
    _default_dataset = str(_project_root / 'dataset' / 'sdata')
    _default_out = str(_project_root / 'embeddings' / 'sdata')

    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default=_default_dataset, help='Path to sdata folder')
    p.add_argument('--out-dir', type=str, default=_default_out, help='Output dir for embeddings')
    p.add_argument('--vision-encoder', choices=['clip', 'viscop'], default='viscop')
    p.add_argument('--audio-encoder', choices=['wav2vec', 'whisper'], default='wav2vec')
    p.add_argument('--viscop-model-path', type=str,
                   default='viscop_trained_models/viscop_qwen2.5_7b_viscop-lora_egocentric-expert')
    p.add_argument('--cross-pair', action='store_true')
    p.add_argument('--augment-variations', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=24, help='Batch size for encoding (try 32–48 on A100 if OOM)')
    p.add_argument('--device', default='cuda:auto' if torch.cuda.is_available() else 'cpu',
                   help='Device: cuda:auto (pick freest GPU), cuda:0..3, cuda, auto (multi-GPU), cpu')
    args = p.parse_args()

    # Resolve device
    if args.device == 'cuda:auto' and torch.cuda.is_available():
        # Pick GPU with most free memory
        best_idx, best_free = 0, 0
        for i in range(torch.cuda.device_count()):
            try:
                if hasattr(torch.cuda, 'mem_get_info'):
                    free, _ = torch.cuda.mem_get_info(i)
                else:
                    free = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                if free > best_free:
                    best_free, best_idx = free, i
            except Exception:
                pass
        device = torch.device(f'cuda:{best_idx}')
        log.info("Using GPU: %s (%s) - %.1f GB free", device, torch.cuda.get_device_name(device), best_free / 1e9)
    elif args.device == 'auto' and torch.cuda.is_available():
        device = torch.device('cuda:0')  # primary; model will use device_map="auto" across all
        log.info("Using multi-GPU (device_map=auto) across %d GPUs", torch.cuda.device_count())
    elif 'cuda' in args.device and not torch.cuda.is_available():
        log.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
        if device.type == 'cuda':
            log.info("Using GPU: %s (%s)", device, torch.cuda.get_device_name(device))
        else:
            log.info("Using CPU (no GPU). For HPC: request GPU in your job, e.g. #SBATCH --gres=gpu:1")

    from mmfuse.preprocessing.preprocessor import VisionPreprocessor, AudioPreprocessor
    from mmfuse.encoders.vision_encoder import VisionEncoder
    from mmfuse.encoders.vision_encoder_viscop import VisCoPVisionEncoder
    from mmfuse.encoders.audio_encoder import Wav2VecPooledEncoder
    from mmfuse.encoders.audio_encoder_whisper import WhisperAudioEncoder

    root = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_sample_list(root, args.cross_pair, args.augment_variations)
    if not samples:
        raise RuntimeError(f"No samples in {args.dataset}")
    log.info("Samples: %d", len(samples))
    vprep = VisionPreprocessor(image_size=(224, 224), normalize=True)
    aprep = AudioPreprocessor(sample_rate=16000, duration=2.5)
    target_audio_samples = int(aprep.duration * aprep.sample_rate)

    # Vision encoder
    use_multi_gpu = (args.device == 'auto' and torch.cuda.is_available())
    if args.vision_encoder == 'viscop':
        vision = VisCoPVisionEncoder(
            model_path=args.viscop_model_path,
            device='auto' if use_multi_gpu else str(device),
        )
        if not use_multi_gpu:
            vision = vision.to(device)
        use_clip_preprocess = False  # VisCoP has its own processor
    else:
        vision = VisionEncoder(device=str(device)).to(device)
        use_clip_preprocess = True

    # Audio encoder
    if args.audio_encoder == 'wav2vec':
        audio = Wav2VecPooledEncoder(frozen=True, device=str(device)).to(device)
    else:
        audio = WhisperAudioEncoder(frozen=True, device=str(device)).to(device)

    vision.eval()
    audio.eval()

    # Save config for training
    import json
    num_classes = len(set(s[3] for s in samples))
    config = {
        'vision_encoder': args.vision_encoder,
        'audio_encoder': args.audio_encoder,
        'vision_dim': vision.output_dim,
        'audio_dim': 768,
        'cross_pair': args.cross_pair,
        'augment_variations': args.augment_variations,
        'num_samples': len(samples),
        'num_classes': num_classes,
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    num_batches = (len(samples) + args.batch_size - 1) // args.batch_size
    log.info("Precomputing %d samples in %d batches (batch_size=%d)...", len(samples), num_batches, args.batch_size)
    t0 = time.time()

    for i in range(0, len(samples), args.batch_size):
        batch = samples[i:i + args.batch_size]
        frames1, frames2, audios, targets = [], [], [], []

        for audio_path, cam1_path, cam2_path, label, v in batch:
            f1 = load_frame(cam1_path)
            f2 = load_frame(cam2_path)
            f1 = _augment_frame(f1, v)
            f2 = _augment_frame(f2, v)
            frames1.append(f1)
            frames2.append(f2)
            targets.append(label)

            if audio_path and Path(audio_path).exists():
                try:
                    a = aprep.preprocess(str(audio_path))
                except Exception:
                    a = torch.zeros(target_audio_samples)
            else:
                a = torch.zeros(target_audio_samples)
            audios.append(a)

        # Vision
        if use_clip_preprocess:
            vis_t1 = torch.stack([vprep.preprocess(f) for f in frames1]).to(device)
            vis_t2 = torch.stack([vprep.preprocess(f) for f in frames2]).to(device)
        else:
            vis_t1 = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames1]).to(device)
            vis_t2 = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames2]).to(device)

        with torch.no_grad():
            v_emb1 = vision(vis_t1)
            v_emb2 = vision(vis_t2)

        # Audio
        audio_batch = torch.stack(audios).to(device)
        with torch.no_grad():
            a_emb = audio(audio_batch)

        for j, (_, _, _, label, _) in enumerate(batch):
            idx = i + j
            torch.save({
                'vision_camera1': v_emb1[j].cpu(),
                'vision_camera2': v_emb2[j].cpu(),
                'audio': a_emb[j].cpu(),
                'target': label,
            }, out_dir / f'{idx:08d}.pt')

        done = min(i + args.batch_size, len(samples))
        batch_num = (i // args.batch_size) + 1
        if batch_num % 50 == 0 or done >= len(samples):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(samples) - done) / rate if rate > 0 else 0
            log.info("Precomputed %d / %d (%.1f%%) | %.1f samples/s | ETA %.0fm",
                     done, len(samples), 100 * done / len(samples), rate, eta / 60)

    elapsed = time.time() - t0
    log.info("Done in %.1fm. Embeddings saved to %s", elapsed / 60, out_dir)


if __name__ == '__main__':
    main()
