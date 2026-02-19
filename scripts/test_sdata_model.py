#!/usr/bin/env python3
"""
Test trained sdata model and print output values (action + movement).
Shows action logits, probabilities, predicted class, and movement vector per sample.

Usage:
  python scripts/test_sdata_model.py --checkpoint runs/sdata_viscop/ckpt_sdata_epoch_10.pt \\
      --embeddings-dir mmfuse/embeddings/sdata_viscop --num-samples 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmfuse.encoders.sensor_encoder import PressureSensorEncoder, EMGSensorEncoder
from mmfuse.fusion.multimodal_fusion import MultimodalFusionWithAttention


def collate_fn(batch):
    return batch


class ActionClassifier(torch.nn.Module):
    def __init__(self, embedding_dim=256, num_classes=8):
        super().__init__()
        self.fc = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MovementHead(torch.nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.fc = torch.nn.Linear(embedding_dim, 3)

    def forward(self, x):
        return self.fc(x)


class PrecomputedSDataDataset(torch.utils.data.Dataset):
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
    v1 = torch.stack([s['vision_camera1'] for s in batch]).to(device).float()
    v2 = torch.stack([s['vision_camera2'] for s in batch]).to(device).float()
    a = torch.stack([s['audio'] for s in batch]).to(device).float()
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
    fused, _ = encoders['fusion'](embeddings, return_kl=True)
    return fused


# Command names for display (adjust to match your part order)
COMMAND_NAMES = [
    "Start the Massage",
    "Focus Here",
    "Move down a little bit",
    "Go Back Up",
    "Stop. Pause for a second",
    "Move to the Left",
    "Move to the Right",
    "Right there, perfect spot",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=None, help='Path to checkpoint (.pt); if missing, auto-find latest')
    p.add_argument('--embeddings-dir', required=True, help='Path to precomputed embeddings')
    p.add_argument('--num-samples', type=int, default=10, help='Number of samples to test')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    ckpt_path = args.checkpoint
    if not ckpt_path or not Path(ckpt_path).exists():
        proj = Path(__file__).resolve().parent.parent
        candidates = sorted(proj.glob('**/ckpt_sdata_epoch_*.pt'))
        if not candidates:
            candidates = sorted(Path('.').glob('**/ckpt_sdata_epoch_*.pt'))
        if candidates:
            ckpt_path = str(candidates[-1])
            print(f"Using checkpoint: {ckpt_path}")
        else:
            print("No checkpoint found. Use --checkpoint path/to/ckpt_sdata_epoch_N.pt")
            sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    num_classes = ckpt['num_classes']
    fusion_dim = ckpt['fusion_dim']
    vision_dim = ckpt.get('vision_dim', 3584)
    audio_dim = 768

    with open(Path(args.embeddings_dir) / 'config.json') as f:
        emb_config = json.load(f)
    vision_dim = emb_config.get('vision_dim', vision_dim)
    num_classes = emb_config.get('num_classes', num_classes)

    ds = PrecomputedSDataDataset(Path(args.embeddings_dir), emb_config)
    np.random.seed(args.seed)
    indices = np.random.choice(len(ds), min(args.num_samples, len(ds)), replace=False)
    dl = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False, collate_fn=collate_fn)

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
        fusion_dim=fusion_dim,
        num_heads=8,
        dropout=0.2,
    ).to(device)
    model = ActionClassifier(embedding_dim=fusion_dim, num_classes=num_classes).to(device)
    movement_head = MovementHead(embedding_dim=fusion_dim).to(device)

    fusion.load_state_dict(ckpt['fusion_state'])
    model.load_state_dict(ckpt['model_state'])
    has_movement = 'movement_state' in ckpt
    if has_movement:
        movement_head.load_state_dict(ckpt['movement_state'])
    movement_targets = ckpt.get('movement_targets')

    encoders = {'pressure': pressure, 'emg': emg, 'fusion': fusion}
    fusion.eval()
    model.eval()
    movement_head.eval()

    print("=" * 80)
    print("SDATA MODEL TEST OUTPUT")
    print("=" * 80)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Movement head: {'Yes' if has_movement else 'No'}")
    print(f"Num classes: {num_classes}")
    print("=" * 80)

    for i, batch in enumerate(dl):
        target = batch[0]['target']
        fused = build_embedding_precomputed(batch, device, encoders)
        with torch.no_grad():
            logits = model(fused)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = logits.argmax(dim=1).item()
            if has_movement:
                movement = movement_head(fused).cpu().numpy()[0]
            else:
                movement = None

        print(f"\n--- Sample {i + 1} ---")
        print(f"  Ground truth:  action={target} ({COMMAND_NAMES[target] if target < len(COMMAND_NAMES) else '?'})")
        print(f"  Predicted:     action={pred} ({COMMAND_NAMES[pred] if pred < len(COMMAND_NAMES) else '?'})")
        print(f"  Correct:       {'Yes' if pred == target else 'No'}")
        print(f"  Action logits: {logits.cpu().numpy()[0].round(3).tolist()}")
        print(f"  Action probs:  {[f'{p:.4f}' for p in probs]}")
        if has_movement:
            print(f"  Movement out:  delta_along={movement[0]:.4f}, delta_lateral={movement[1]:.4f}, magnitude={movement[2]:.4f}")
            if movement_targets is not None and target < movement_targets.shape[0]:
                exp = movement_targets[target].cpu().numpy()
                print(f"  Expected:      delta_along={exp[0]:.4f}, delta_lateral={exp[1]:.4f}, magnitude={exp[2]:.4f}")
        print()

    print("=" * 80)


if __name__ == '__main__':
    main()
