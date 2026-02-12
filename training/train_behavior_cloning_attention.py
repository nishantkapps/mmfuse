#!/usr/bin/env python3
"""
Behavior cloning training with MultimodalFusionWithAttention

Trains the full pipeline (fusion + decoder) using:
- Cross-modal attention over projected embeddings
- KL divergence for knowledge distillation (camera-camera alignment)
- RoboticArmController3DOF decoder

Same dataset format as train_behavior_cloning.py:
dataset/<volunteer>/<session>/<trial>/ with cam1.mp4, cam2.mp4 (optional),
mic_speech.wav, robot.csv, pressure.csv, emg.csv, meta.json
"""
import argparse
import os
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

from mmfuse.preprocessing.preprocessor import VisionPreprocessor, AudioPreprocessor
from mmfuse.encoders.vision_encoder import VisionEncoder
from mmfuse.encoders.audio_encoder_learnable import AudioEncoder as LearnableAudioEncoder
from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention
from mmfuse.ctrl.robotic_arm_controller import RoboticArmController3DOF


class TrialDataset(Dataset):
    def __init__(self, root_dir, device='cpu'):
        self.root_dir = Path(root_dir)
        self.trials = [p.parent for p in self.root_dir.rglob('meta.json')]
        self.device = device
        self.vprep = VisionPreprocessor(image_size=(224, 224))
        self.aprep = AudioPreprocessor(sample_rate=16000, duration=2.5)

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        tdir = self.trials[idx]
        # cam1
        cam1 = tdir / 'cam1.mp4'
        if cam1.exists():
            cap = cv2.VideoCapture(str(cam1))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            mid = max(0, n // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # cam2
        cam2 = tdir / 'cam2.mp4'
        frame2 = None
        if cam2.exists():
            cap2 = cv2.VideoCapture(str(cam2))
            n2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            mid2 = max(0, n2 // 2)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, mid2)
            ret2, frame2 = cap2.read()
            cap2.release()
            if ret2 and frame2 is not None:
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            else:
                frame2 = None

        audio_path = tdir / 'mic_speech.wav'
        if not audio_path.exists():
            audio_path = tdir / 'mic.wav'

        robot_csv = tdir / 'robot.csv'
        target = np.zeros(4, dtype=np.float32)
        if robot_csv.exists():
            with open(robot_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if len(rows) > 0:
                    midr = rows[len(rows)//2]
                    try:
                        target[0] = float(midr.get('x', 0.0))
                        target[1] = float(midr.get('y', 0.0))
                        target[2] = float(midr.get('z', 0.0))
                        target[3] = float(midr.get('force', 0.0))
                    except Exception:
                        pass

        return {
            'frame': frame,
            'frame2': frame2,
            'audio': str(audio_path) if audio_path.exists() else None,
            'pressure': str(tdir / 'pressure.csv') if (tdir / 'pressure.csv').exists() else None,
            'emg': str(tdir / 'emg.csv') if (tdir / 'emg.csv').exists() else None,
            'target': target
        }


def collate_fn(batch):
    return batch


def build_embedding(batch, device, encoders):
    """
    Compute fused embedding and KL losses using MultimodalFusionWithAttention.
    Uses vision_camera1, vision_camera2 (or duplicate when single cam), audio, pressure, emg.
    """
    vis_imgs = []
    vis_imgs2 = []
    auds = []
    pressures = []
    emgs = []
    for s in batch:
        vis_imgs.append(s['frame'])
        vis_imgs2.append(s.get('frame2', None))
        auds.append(s['audio'])
        if s['pressure'] and Path(s['pressure']).exists():
            pressures.append(np.loadtxt(s['pressure'], delimiter=',', skiprows=1))
        else:
            pressures.append(np.zeros((1,)))
        if s['emg'] and Path(s['emg']).exists():
            emgs.append(np.loadtxt(s['emg'], delimiter=',', skiprows=1))
        else:
            emgs.append(np.zeros((1,)))

    vprep = VisionPreprocessor()
    vis_tensors = [vprep.preprocess(img) for img in vis_imgs]
    vis_batch = torch.stack(vis_tensors).to(device)

    has_cam2 = any(f is not None for f in vis_imgs2)
    if has_cam2:
        vis2_imgs = [img if img is not None else np.zeros((224, 224, 3), dtype=np.uint8) for img in vis_imgs2]
        vis2_tensors = [vprep.preprocess(img) for img in vis2_imgs]
        vis2_batch = torch.stack(vis2_tensors).to(device)
    else:
        vis2_batch = vis_batch  # duplicate when single cam

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


def train(args):
    device = torch.device(args.device)

    ds = TrialDataset(args.dataset, device=str(device))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    vision = VisionEncoder(device=str(device)).to(device)
    audio = LearnableAudioEncoder(device=str(device)).to(device)
    pressure = PressureSensorEncoder(output_dim=256, input_features=2).to(device)
    emg = EMGSensorEncoder(output_dim=256, num_channels=3, input_features=4).to(device)

    # MultimodalFusionWithAttention: 5 modalities for cross-modal attention + KL distillation
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

    model = RoboticArmController3DOF(embedding_dim=512, device=str(device)).to(device)
    trainable_params = list(model.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    criterion = torch.nn.MSELoss()

    encoders = {'vision': vision, 'audio': audio, 'pressure': pressure, 'emg': emg, 'fusion': fusion}

    # Print architecture
    print("\n" + "=" * 60)
    print("MODEL: MultimodalFusionWithAttention + RoboticArmController3DOF")
    print("=" * 60)
    print("\n--- Fusion pipeline (forward flow)")
    print("  1. Projections:  modality_emb -> Linear+BN+ReLU -> fusion_dim (per modality)")
    print("  2. KL Divergence (knowledge distillation):")
    print("       - vision_camera1 <-> vision_camera2  (camera alignment)")
    print("       - (text <-> audio when available)")
    print("  3. Cross-Modal Attention: MultiheadAttention (num_heads={})".format(args.num_heads))
    print("  4. Fusion MLP: concat(attended) -> Linear -> ReLU -> Dropout -> Linear -> fused")
    print("\n--- Fusion submodules")
    print(fusion)
    print("\n--- Decoder")
    print(model)

    def _count(m):
        return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)

    components = [
        ("vision (CLIP)", vision),
        ("audio", audio),
        ("pressure", pressure),
        ("emg", emg),
        ("fusion (attention + KL)", fusion),
        ("decoder", model),
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
        count = 0
        for batch in dl:
            targets = torch.stack([torch.tensor(s['target'], dtype=torch.float32) for s in batch]).to(device)

            fused, kl_losses = build_embedding(batch, device, encoders)
            out = model.decode(fused)
            pred = torch.cat([out['position'], out['force']], dim=1)

            loss_bc = criterion(pred, targets)
            loss_kl = sum(kl_losses.values()) if kl_losses else torch.tensor(0.0, device=device)
            loss = loss_bc + args.kl_weight * loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss_bc.item()
            if kl_losses:
                epoch_kl += sum(v.item() for v in kl_losses.values())
            count += 1

        kl_str = f" | KL={epoch_kl/count:.4f}" if len(kl_losses) > 0 else ""
        print(f"Epoch {epoch+1}/{args.epochs} loss={epoch_loss/count:.6f}{kl_str}")

        ckpt = {
            'model_state': model.state_dict(),
            'fusion_state': fusion.state_dict(),
            'epoch': epoch + 1,
        }
        torch.save(ckpt, os.path.join(args.out_dir, f'ckpt_attention_epoch_{epoch+1}.pt'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Root dataset folder')
    p.add_argument('--out-dir', default='checkpoints', help='Where to save checkpoints')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--freeze-encoders', action='store_true')
    p.add_argument('--kl-weight', type=float, default=0.1, help='Weight for KL distillation loss')
    p.add_argument('--num-heads', type=int, default=8, help='Attention heads in fusion')
    p.add_argument('--dropout', type=float, default=0.2)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
