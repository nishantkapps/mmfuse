#!/usr/bin/env python3
"""
Behavior cloning training scaffold

Trains `RoboticArmController3DOF` to map fused multimodal embeddings
to robot commands using collected demonstration trials.

This script expects dataset folders structured like:
dataset/<volunteer>/<session>/<trial>/ with files:
 - cam1.mp4
 - mic_speech.wav (or mic.wav)
 - robot.csv (with columns x,y,z,force) OR joint values (fallback)
 - pressure.csv, emg.csv
 - meta.json

The script uses the existing encoders/fusion modules to compute embeddings.
"""
import argparse
import os
import json
import glob
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

from preprocessing.preprocessor import VisionPreprocessor, AudioPreprocessor, SensorPreprocessor
from encoders.vision_encoder import VisionEncoder
from encoders.audio_encoder_learnable import AudioEncoder as LearnableAudioEncoder
from encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
from fusion.multimodal_fusion import MultimodalFusion
from ctrl.robotic_arm_controller import RoboticArmController3DOF


class TrialDataset(Dataset):
    def __init__(self, root_dir, device='cpu'):
        self.root_dir = Path(root_dir)
        # find trial folders (one level deep or deeper)
        self.trials = [p for p in self.root_dir.rglob('meta.json')]
        self.trials = [p.parent for p in self.trials]
        self.device = device

        # preprocessors
        self.vprep = VisionPreprocessor(image_size=(224, 224))
        self.aprep = AudioPreprocessor(sample_rate=16000, duration=2.5)
        self.sprep = SensorPreprocessor()

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        tdir = self.trials[idx]
        # load middle frame from cam1
        cam1 = tdir / 'cam1.mp4'
        if cam1.exists():
            cap = cv2.VideoCapture(str(cam1))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            mid = max(0, n // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # load middle frame from cam2 if present
        cam2 = tdir / 'cam2.mp4'
        if cam2.exists():
            cap2 = cv2.VideoCapture(str(cam2))
            n2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            mid2 = max(0, n2 // 2)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, mid2)
            ret2, frame2 = cap2.read()
            cap2.release()
            if not ret2:
                frame2 = None
        else:
            frame2 = None

        # audio
        audio_path = tdir / 'mic_speech.wav'
        if not audio_path.exists():
            audio_path = tdir / 'mic.wav'

        # robot targets
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
                        # fallback to zeros
                        pass

        sample = {
            'frame': frame,
            'frame2': frame2,
            'audio': str(audio_path) if audio_path.exists() else None,
            'pressure': str(tdir / 'pressure.csv') if (tdir / 'pressure.csv').exists() else None,
            'emg': str(tdir / 'emg.csv') if (tdir / 'emg.csv').exists() else None,
            'target': target
        }

        return sample


def collate_fn(batch):
    return batch


def build_embedding(batch, device, encoders):
    """Compute fused embedding for a batch (list of samples)

    encoders: dict with `vision`, `audio`, `pressure`, `emg`, `fusion`
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
        # sensors: load small arrays if present
        if s['pressure'] and Path(s['pressure']).exists():
            pressures.append(np.loadtxt(s['pressure'], delimiter=',', skiprows=1))
        else:
            pressures.append(np.zeros((1,)))
        if s['emg'] and Path(s['emg']).exists():
            emgs.append(np.loadtxt(s['emg'], delimiter=',', skiprows=1))
        else:
            emgs.append(np.zeros((1,)))

    # preprocess vision (handle optional second camera)
    vprep = VisionPreprocessor()
    vis_tensors = [vprep.preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in vis_imgs]
    vis_batch = torch.stack(vis_tensors).to(device)
    # second camera batch (if available)
    has_cam2 = any(f is not None for f in vis_imgs2)
    if has_cam2:
        vis2_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((224,224,3),dtype=np.uint8) for img in vis_imgs2]
        vis2_tensors = [vprep.preprocess(img) for img in vis2_imgs]
        vis2_batch = torch.stack(vis2_tensors).to(device)
    else:
        vis2_batch = None

    # audio preprocess and encode
    aprep = AudioPreprocessor()
    audio_tensors = []
    for a in auds:
        if a is None:
            audio_tensors.append(torch.zeros(int(2.5 * 16000)))
        else:
            audio_tensors.append(aprep.preprocess(a))
    audio_batch = torch.stack(audio_tensors).to(device)

    # sensor preprocessing: use mean as simple feature
    pressure_feats = [torch.tensor(p.mean(axis=0) if p.ndim>1 else p.mean()) for p in pressures]
    emg_feats = [torch.tensor(e.mean(axis=0) if e.ndim>1 else e.mean()) for e in emgs]
    pressure_batch = torch.stack([f.float() for f in pressure_feats]).to(device)
    emg_batch = torch.stack([f.float() for f in emg_feats]).to(device)

    # encode
    with torch.no_grad():
        v_emb = encoders['vision'](vis_batch)
        if vis2_batch is not None:
            v_emb2 = encoders['vision'](vis2_batch)
            # combine embeddings from both cameras (average)
            v_emb = (v_emb + v_emb2) / 2.0
        a_emb = encoders['audio'](audio_batch.unsqueeze(1)) if audio_batch.dim()==2 else encoders['audio'](audio_batch)
        p_emb = encoders['pressure'](pressure_batch.unsqueeze(0)) if pressure_batch.dim()==1 else encoders['pressure'](pressure_batch)
        e_emb = encoders['emg'](emg_batch.unsqueeze(0)) if emg_batch.dim()==1 else encoders['emg'](emg_batch)

    # fuse
    fused = encoders['fusion']({ 'vision': v_emb, 'audio': a_emb, 'pressure': p_emb, 'emg': e_emb })
    return fused


def train(args):
    device = torch.device(args.device)

    # dataset
    ds = TrialDataset(args.dataset, device=str(device))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # encoders (frozen by default) — use a single shared vision encoder to save parameters
    vision = VisionEncoder(device=str(device)).to(device)
    audio = LearnableAudioEncoder(device=str(device)).to(device)
    pressure = PressureSensorEncoder(output_dim=256).to(device)
    emg = EMGSensorEncoder(output_dim=256, num_channels=3).to(device)
    fusion = MultimodalFusion(modality_dims={'vision':vision.output_dim,'audio':768,'pressure':256,'emg':256}, fusion_dim=512).to(device)

    if args.freeze_encoders:
        for p in list(vision.parameters())+list(audio.parameters())+list(pressure.parameters())+list(emg.parameters())+list(fusion.parameters()):
            p.requires_grad = False

    # model to train: RoboticArmController3DOF (decoder)
    model = RoboticArmController3DOF(embedding_dim=512, device=str(device)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    encoders = {'vision': vision, 'audio':audio, 'pressure':pressure, 'emg':emg, 'fusion':fusion}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        count = 0
        for batch in dl:
            targets = []
            for s in batch:
                targets.append(torch.tensor(s['target'], dtype=torch.float32))
            targets = torch.stack(targets).to(device)

            fused = build_embedding(batch, device, encoders)
            out = model.decode(fused)
            pred = torch.cat([out['position'], out['force']], dim=1)

            loss = criterion(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            count += 1

        print(f"Epoch {epoch+1}/{args.epochs} loss={epoch_loss/count:.6f}")
        # save checkpoint
        ckpt = {'model_state': model.state_dict(), 'epoch': epoch+1}
        torch.save(ckpt, os.path.join(args.out_dir, f'ckpt_epoch_{epoch+1}.pt'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Root dataset folder')
    p.add_argument('--out-dir', default='checkpoints', help='Where to save checkpoints')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--freeze-encoders', action='store_true')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
