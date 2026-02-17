#!/usr/bin/env python3
"""
Train MultimodalFusionWithAttention on sdata folder structure.

Dataset layout: dataset/sdata/
  part1/   <- action 0
    p005/
      p005_c1_part1.mp4   (camera 1)
      p005_c2_part1.mp4   (camera 2)
      [audio files to be added later]
    p010/
      ...
  part2/   <- action 1
    ...
  partN/   <- action N-1

Each part = different action. Participant folders contain 2 videos (cam1, cam2).
Audio encoder is wired; add audio files to participant folders when ready.
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

from mmfuse.preprocessing.preprocessor import VisionPreprocessor, AudioPreprocessor
from mmfuse.encoders.vision_encoder import VisionEncoder
from mmfuse.encoders.audio_encoder_learnable import AudioEncoder as LearnableAudioEncoder
from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention


class SDataDataset(Dataset):
    """
    Loads from sdata/partX/participant_id/ with *_c1_*.mp4 and *_c2_*.mp4.
    Label = part index (action class). Audio: placeholder until files added.
    """
    def __init__(self, root_dir, device='cpu'):
        self.root_dir = Path(root_dir)
        self.device = device
        self.vprep = VisionPreprocessor(image_size=(224, 224))
        self.aprep = AudioPreprocessor(sample_rate=16000, duration=2.5)

        # Collect (participant_dir, action_label) for each part
        self.samples = []
        part_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('part')])
        self.part_to_label = {p.name: i for i, p in enumerate(part_dirs)}
        self.num_classes = len(part_dirs)

        for part_dir in part_dirs:
            label = self.part_to_label[part_dir.name]
            for pdir in part_dir.iterdir():
                if not pdir.is_dir():
                    continue
                cam1 = list(pdir.glob('*_c1_*.mp4'))
                cam2 = list(pdir.glob('*_c2_*.mp4'))
                if cam1 and cam2:
                    self.samples.append((pdir, label, cam1[0], cam2[0]))

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
        pdir, label, cam1_path, cam2_path = self.samples[idx]
        frame1 = self._load_frame(cam1_path)
        frame2 = self._load_frame(cam2_path)

        # Audio: placeholder until files added (mic.wav, mic_speech.wav, etc.)
        audio_path = pdir / 'mic_speech.wav'
        if not audio_path.exists():
            audio_path = pdir / 'mic.wav'
        audio_str = str(audio_path) if audio_path.exists() else None

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
    audio_tensors = []
    for a in auds:
        if a is None:
            audio_tensors.append(torch.zeros(int(2.5 * 16000)))
        else:
            audio_tensors.append(aprep.preprocess(a))
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

    ds = SDataDataset(args.dataset, device=str(device))
    if len(ds) == 0:
        raise RuntimeError(f"No samples found in {args.dataset}. Expected partX/participant_id/*_c1_*.mp4, *_c2_*.mp4")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    num_classes = ds.num_classes
    print(f"Found {len(ds)} samples, {num_classes} action classes")

    vision = VisionEncoder(device=str(device)).to(device)
    audio = LearnableAudioEncoder(device=str(device)).to(device)
    pressure = PressureSensorEncoder(output_dim=256, input_features=2).to(device)
    emg = EMGSensorEncoder(output_dim=256, num_channels=3, input_features=4).to(device)

    modality_dims = {
        'vision_camera1': vision.output_dim,
        'vision_camera2': vision.output_dim,
        'audio': 768,
        'pressure': 256,
        'emg': 256,
    }
    fusion = MultimodalFusionWithAttention(
        modality_dims=modality_dims,
        fusion_dim=512,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    if args.freeze_encoders:
        for m in [vision, audio, pressure, emg]:
            for p in m.parameters():
                p.requires_grad = False

    model = ActionClassifier(embedding_dim=512, num_classes=num_classes).to(device)
    trainable_params = list(model.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    encoders = {'vision': vision, 'audio': audio, 'pressure': pressure, 'emg': emg, 'fusion': fusion}

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
        ("vision (CLIP)", vision),
        ("audio", audio),
        ("pressure", pressure),
        ("emg", emg),
        ("fusion (attention + KL)", fusion),
        ("classifier", model),
    ]
    print("\n--- Parameter counts")
    total_all, trainable_all = 0, 0
    for name, m in components:
        t, tr = _count(m)
        total_all += t
        trainable_all += tr
        status = "trainable" if tr > 0 else "frozen"
        print(f"  {name}: {t:,} ({status})")
    print(f"\n  TOTAL: {total_all:,} | Trainable: {trainable_all:,}")
    print("=" * 60 + "\n")

    for epoch in range(args.epochs):
        model.train()
        fusion.train()
        epoch_loss = 0.0
        epoch_kl = 0.0
        correct = 0
        total = 0
        kl_losses = {}

        for batch in dl:
            targets = torch.tensor([s['target'] for s in batch], dtype=torch.long).to(device)

            fused, kl_losses = build_embedding(batch, device, encoders)
            logits = model(fused)

            loss_bc = criterion(logits, targets)
            loss_kl = sum(kl_losses.values()) if kl_losses else torch.tensor(0.0, device=device)
            loss = loss_bc + args.kl_weight * loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss_bc.item()
            if kl_losses:
                epoch_kl += sum(v.item() for v in kl_losses.values())
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += len(batch)

        count = len(dl)
        acc = correct / total if total > 0 else 0
        kl_str = f" | KL={epoch_kl/count:.4f}" if kl_losses else ""
        print(f"Epoch {epoch+1}/{args.epochs} loss={epoch_loss/count:.4f} acc={acc:.4f}{kl_str}")

        ckpt = {
            'model_state': model.state_dict(),
            'fusion_state': fusion.state_dict(),
            'num_classes': num_classes,
            'epoch': epoch + 1,
        }
        torch.save(ckpt, os.path.join(args.out_dir, f'ckpt_sdata_epoch_{epoch+1}.pt'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='/home/nishant/projects/mmfuse/dataset/sdata', help='Path to sdata folder')
    p.add_argument('--out-dir', default='checkpoints', help='Where to save checkpoints')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--freeze-encoders', action='store_true')
    p.add_argument('--kl-weight', type=float, default=0.1)
    p.add_argument('--num-heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.2)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
