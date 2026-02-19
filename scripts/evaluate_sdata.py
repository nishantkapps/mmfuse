#!/usr/bin/env python3
"""
Evaluate trained sdata model and generate paper-ready visualizations:
- Test accuracy summary
- Confusion matrix
- ROC curves (one-vs-rest, with AUC)
- Per-class precision, recall, F1
- Training curves (if history JSON exists)

Usage:
  python scripts/evaluate_sdata.py --checkpoint runs/sdata_viscop/ckpt_sdata_epoch_10.pt \\
      --embeddings-dir mmfuse/embeddings/sdata_viscop --out-dir runs/sdata_viscop/figures
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
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


def run_inference(model, fusion, test_dl, device, encoders):
    """Run inference and return y_true, y_pred, y_proba."""
    model.eval()
    fusion.eval()
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            targets = torch.tensor([s['target'] for s in batch], dtype=torch.long).to(device)
            fused = build_embedding_precomputed(batch, device, encoders)
            logits = model(fused)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_proba.extend(probs.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_proba)


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted',
           ylabel='True')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_true, y_proba, class_names, out_path):
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = []
    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        if len(np.unique(y_binary)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
        auc = roc_auc_score(y_binary, y_proba[:, i])
        aucs.append(auc)
        ax.plot(fpr, tpr, label=f'{class_names[i]} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    macro_auc = np.mean(aucs) if aucs else 0.0
    ax.set_title(f'ROC Curves (One-vs-Rest) | Macro AUC={macro_auc:.3f}')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return macro_auc


def plot_training_curves(history_path, out_path):
    if not Path(history_path).exists():
        return
    with open(history_path) as f:
        history = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = list(range(1, len(history.get('train_loss', [])) + 1))
    axes[0].plot(epochs, history.get('train_loss', []), 'b-', label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history.get('train_acc', []), 'b-', label='Train Acc')
    axes[1].plot(epochs, history.get('test_acc', []), 'g-', label='Test Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to checkpoint (.pt)')
    p.add_argument('--embeddings-dir', required=True, help='Path to precomputed embeddings')
    p.add_argument('--out-dir', default=None, help='Output dir for figures (default: same as checkpoint dir)')
    p.add_argument('--class-names', nargs='+', default=None,
                   help='Class names (default: part0, part1, ...)')
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    emb_dir = Path(args.embeddings_dir)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    num_classes = ckpt['num_classes']
    fusion_dim = ckpt['fusion_dim']
    vision_dim = ckpt.get('vision_dim', 3584)  # VisCoP default
    audio_dim = 768

    with open(emb_dir / 'config.json') as f:
        emb_config = json.load(f)
    vision_dim = emb_config.get('vision_dim', vision_dim)
    num_classes = emb_config.get('num_classes', num_classes)

    ds = PrecomputedSDataDataset(emb_dir, emb_config)
    labels = [torch.load(p, map_location='cpu', weights_only=True)['target'] for p in ds.samples]
    train_idx, test_idx = train_test_split(
        range(len(ds)), test_size=0.1, stratify=labels, random_state=42
    )
    test_ds = Subset(ds, test_idx)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

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

    fusion.load_state_dict(ckpt['fusion_state'])
    model.load_state_dict(ckpt['model_state'])
    encoders = {'pressure': pressure, 'emg': emg, 'fusion': fusion}

    y_true, y_pred, y_proba = run_inference(model, fusion, test_dl, device, encoders)

    accuracy = accuracy_score(y_true, y_pred)
    class_names = args.class_names or [f'part{i}' for i in range(num_classes)]
    try:
        macro_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    except Exception:
        macro_auc = 0.0

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro AUC (One-vs-Rest): {macro_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({
            'accuracy': float(accuracy),
            'macro_auc': float(macro_auc),
            'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }, f, indent=2)

    plot_confusion_matrix(y_true, y_pred, class_names, out_dir / 'confusion_matrix.png')
    plot_roc_curves(y_true, y_proba, class_names, out_dir / 'roc_curves.png')

    history_path = ckpt_path.parent / 'training_history.json'
    if history_path.exists():
        plot_training_curves(history_path, out_dir / 'training_curves.png')

    print(f"\nFigures saved to {out_dir}")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curves.png")
    print(f"  - metrics.json")
    if history_path.exists():
        print(f"  - training_curves.png")


if __name__ == '__main__':
    main()
