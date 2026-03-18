#!/usr/bin/env python3
"""
MIRA Challenge — Official Evaluation Script
Evaluates participant submissions for Track 1 (Action Recognition) and Track 2 (Motion Prediction).

Usage:
  python competition/evaluate.py \
      --ground-truth competition/ground_truth.csv \
      --predictions submission.csv \
      --output results.json

Ground-truth CSV format:
  sample,action,dx,dy

Submission CSV format (participants produce this):
  sample,action,dx,dy

The script computes:
  - Track 1: Macro F1 (primary), Accuracy, per-class Recall, Modality Robustness Score
  - Track 2: MAE (primary)
  - Combined Score: 0.60 * Macro-F1 + 0.40 * (1 - NormalizedMAE)
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import numpy as np


def load_csv(path):
    """Load CSV into list of dicts. Expects columns: sample, action, dx, dy."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'sample': r['sample'].strip(),
                'action': r['action'].strip(),
                'dx': float(r['dx']),
                'dy': float(r['dy']),
            })
    return {r['sample']: r for r in rows}


def evaluate_track1(gt, pred, label_names=None):
    """Track 1: Action Recognition. Returns dict of metrics."""
    samples = sorted(gt.keys() & pred.keys())
    if not samples:
        return {'error': 'No matching samples'}

    y_true = [gt[s]['action'] for s in samples]
    y_pred = [pred[s]['action'] for s in samples]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    if label_names is None:
        label_names = sorted(set(y_true))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=label_names, average=None, zero_division=0)
    per_class = {
        label_names[i]: {'precision': float(prec[i]), 'recall': float(rec[i]),
                         'f1': float(f1[i]), 'support': int(sup[i])}
        for i in range(len(label_names))
    }

    cm = confusion_matrix(y_true, y_pred, labels=label_names)

    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'num_samples': len(samples),
    }


def evaluate_track1_robustness(gt_full, gt_missing, pred_full, pred_missing):
    """Compute Modality Robustness Score = F1_missing / F1_full."""
    r_full = evaluate_track1(gt_full, pred_full)
    r_missing = evaluate_track1(gt_missing, pred_missing)
    f1_full = r_full.get('macro_f1', 0)
    f1_missing = r_missing.get('macro_f1', 0)
    mrs = f1_missing / f1_full if f1_full > 0 else 0.0
    return {
        'f1_full': f1_full,
        'f1_missing_modality': f1_missing,
        'modality_robustness_score': float(mrs),
    }


def evaluate_track2(gt, pred):
    """Track 2: Motion Prediction. Returns MAE and per-axis MAE."""
    samples = sorted(gt.keys() & pred.keys())
    if not samples:
        return {'error': 'No matching samples'}

    errors = []
    errors_x, errors_y = [], []
    for s in samples:
        dx_err = abs(gt[s]['dx'] - pred[s]['dx'])
        dy_err = abs(gt[s]['dy'] - pred[s]['dy'])
        errors_x.append(dx_err)
        errors_y.append(dy_err)
        errors.append(dx_err + dy_err)

    mae = float(np.mean(errors))
    mae_x = float(np.mean(errors_x))
    mae_y = float(np.mean(errors_y))
    rmse = float(np.sqrt(np.mean([e**2 for e in errors])))

    return {
        'mae': mae,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'rmse': rmse,
        'num_samples': len(samples),
    }


def combined_score(macro_f1, mae, max_mae):
    """Combined leaderboard score for Tracks 1+2."""
    normalized_mae = min(mae / max_mae, 1.0) if max_mae > 0 else 0.0
    return 0.60 * macro_f1 + 0.40 * (1.0 - normalized_mae)


def main():
    parser = argparse.ArgumentParser(description='MIRA Challenge Evaluation')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth CSV')
    parser.add_argument('--predictions', required=True, help='Path to submission CSV')
    parser.add_argument('--output', default='results.json', help='Output JSON path')
    parser.add_argument('--max-mae', type=float, default=20.0,
                        help='Max MAE for normalization (cm). Default 20.0')
    parser.add_argument('--ground-truth-missing', default=None,
                        help='Ground truth CSV for missing-modality subset (optional)')
    parser.add_argument('--predictions-missing', default=None,
                        help='Predictions CSV for missing-modality subset (optional)')
    args = parser.parse_args()

    gt = load_csv(args.ground_truth)
    pred = load_csv(args.predictions)

    label_names = ['Start', 'Go Here', 'Move Down', 'Move Up',
                   'Stop', 'Move Left', 'Move Right', 'Perfect']

    track1 = evaluate_track1(gt, pred, label_names)
    track2 = evaluate_track2(gt, pred)

    robustness = {}
    if args.ground_truth_missing and args.predictions_missing:
        gt_m = load_csv(args.ground_truth_missing)
        pred_m = load_csv(args.predictions_missing)
        robustness = evaluate_track1_robustness(gt, gt_m, pred, pred_m)

    score = combined_score(track1['macro_f1'], track2['mae'], args.max_mae)

    results = {
        'combined_score': float(score),
        'track1_action_recognition': track1,
        'track2_motion_prediction': track2,
    }
    if robustness:
        results['modality_robustness'] = robustness

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"{'='*50}")
    print(f"MIRA Challenge — Evaluation Results")
    print(f"{'='*50}")
    print(f"Track 1 — Macro F1:  {track1['macro_f1']:.4f}")
    print(f"Track 1 — Accuracy:  {track1['accuracy']:.4f}")
    if robustness:
        print(f"Track 1 — MRS:       {robustness['modality_robustness_score']:.4f}")
    print(f"Track 2 — MAE:       {track2['mae']:.4f} cm")
    print(f"Track 2 — RMSE:      {track2['rmse']:.4f} cm")
    print(f"{'='*50}")
    print(f"Combined Score:      {score:.4f}")
    print(f"{'='*50}")
    print(f"\nPer-class breakdown:")
    for cls in label_names:
        if cls in track1['per_class']:
            m = track1['per_class'][cls]
            print(f"  {cls:12s}  prec={m['precision']:.3f}  rec={m['recall']:.3f}  f1={m['f1']:.3f}  n={m['support']}")
    print(f"\nFull results written to {args.output}")


if __name__ == '__main__':
    main()
