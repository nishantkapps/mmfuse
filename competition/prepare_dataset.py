#!/usr/bin/env python3
"""
MIRA Challenge — Dataset Preparation
Generates train/test splits and ground-truth CSV files from the sdata directory.

Split strategy:
  - Train: participants p005-p044 (by sorted order, first ~75%)
  - Test:  participants p045-p065 (held-out, unseen individuals)

Also generates a missing-modality test subset (20% of test, one random modality zeroed).

Usage:
  python competition/prepare_dataset.py \
      --dataset dataset/sdata \
      --output-dir competition/data
"""

import argparse
import csv
import os
import random
from pathlib import Path


ACTION_LABELS = {
    'part1': 'Start',
    'part2': 'Go Here',
    'part3': 'Move Down',
    'part4': 'Move Up',
    'part5': 'Stop',
    'part6': 'Move Left',
    'part7': 'Move Right',
    'part8': 'Perfect',
}


def collect_samples(dataset_root):
    """Collect all (sample_id, participant, part, action, c1, c2, audio) tuples."""
    root = Path(dataset_root)
    samples = []
    for part_dir in sorted(root.iterdir()):
        if not part_dir.is_dir() or not part_dir.name.startswith('part'):
            continue
        action = ACTION_LABELS.get(part_dir.name)
        if action is None:
            continue
        for pdir in sorted(part_dir.iterdir()):
            if not pdir.is_dir():
                continue
            participant = pdir.name
            wavs = sorted(pdir.glob('*_m1_*.wav'))
            for wav in wavs:
                name = wav.stem
                c1 = wav.parent / (name.replace('_m1_', '_c1_') + '.mp4')
                c2 = wav.parent / (name.replace('_m1_', '_c2_') + '.mp4')
                if c1.exists() and c2.exists():
                    samples.append({
                        'sample': wav.name,
                        'participant': participant,
                        'part': part_dir.name,
                        'action': action,
                        'c1': str(c1.relative_to(root)),
                        'c2': str(c2.relative_to(root)),
                        'audio': str(wav.relative_to(root)),
                    })
    return samples


def split_by_participant(samples, test_participants=None):
    """Split samples by participant identity."""
    all_participants = sorted(set(s['participant'] for s in samples))

    if test_participants is None:
        n_train = int(len(all_participants) * 0.75)
        train_participants = set(all_participants[:n_train])
        test_participants = set(all_participants[n_train:])
    else:
        test_participants = set(test_participants)
        train_participants = set(all_participants) - test_participants

    train = [s for s in samples if s['participant'] in train_participants]
    test = [s for s in samples if s['participant'] in test_participants]

    return train, test, sorted(train_participants), sorted(test_participants)


def generate_missing_modality_subset(test_samples, fraction=0.20, seed=42):
    """Select fraction of test samples and assign a random missing modality."""
    rng = random.Random(seed)
    n = max(1, int(len(test_samples) * fraction))
    selected = rng.sample(test_samples, n)
    modalities = ['camera1', 'camera2', 'audio']
    missing = []
    for s in selected:
        s_copy = dict(s)
        s_copy['missing_modality'] = rng.choice(modalities)
        missing.append(s_copy)
    return missing


def write_ground_truth(samples, path, include_displacement=True):
    """Write ground-truth CSV. dx/dy are placeholders (0.0) until MediaPipe extraction."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'action', 'dx', 'dy'])
        for s in samples:
            dx = s.get('dx', 0.0)
            dy = s.get('dy', 0.0)
            w.writerow([s['sample'], s['action'], f"{dx:.2f}", f"{dy:.2f}"])


def write_sample_list(samples, path):
    """Write full sample list with file paths for participants."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'participant', 'part', 'action', 'c1', 'c2', 'audio'])
        for s in samples:
            w.writerow([s['sample'], s['participant'], s['part'],
                        s['action'], s['c1'], s['c2'], s['audio']])


def write_test_samples(samples, path):
    """Write test sample list WITHOUT labels (what participants receive)."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'c1', 'c2', 'audio'])
        for s in samples:
            w.writerow([s['sample'], s['c1'], s['c2'], s['audio']])


def write_missing_modality_list(samples, path):
    """Write missing-modality test list."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'action', 'dx', 'dy', 'missing_modality'])
        for s in samples:
            w.writerow([s['sample'], s['action'],
                        f"{s.get('dx', 0.0):.2f}", f"{s.get('dy', 0.0):.2f}",
                        s['missing_modality']])


def main():
    parser = argparse.ArgumentParser(description='MIRA Challenge — Prepare Dataset')
    parser.add_argument('--dataset', default='dataset/sdata', help='Path to sdata root')
    parser.add_argument('--output-dir', default='competition/data', help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(args.dataset)
    print(f"Collected {len(samples)} samples from {args.dataset}")

    all_participants = sorted(set(s['participant'] for s in samples))
    print(f"Participants: {len(all_participants)} ({all_participants[0]} to {all_participants[-1]})")

    train, test, train_p, test_p = split_by_participant(samples)
    print(f"Train: {len(train)} samples from {len(train_p)} participants ({train_p[0]}..{train_p[-1]})")
    print(f"Test:  {len(test)} samples from {len(test_p)} participants ({test_p[0]}..{test_p[-1]})")

    # Per-class distribution
    from collections import Counter
    train_dist = Counter(s['action'] for s in train)
    test_dist = Counter(s['action'] for s in test)
    print(f"\nPer-class distribution:")
    print(f"  {'Action':12s} {'Train':>6s} {'Test':>6s}")
    for action in ACTION_LABELS.values():
        print(f"  {action:12s} {train_dist[action]:6d} {test_dist[action]:6d}")

    missing = generate_missing_modality_subset(test, seed=args.seed)
    print(f"\nMissing-modality test subset: {len(missing)} samples")

    write_sample_list(train, out / 'train_samples.csv')
    write_ground_truth(train, out / 'train_ground_truth.csv')
    write_test_samples(test, out / 'test_samples.csv')
    write_ground_truth(test, out / 'test_ground_truth.csv')
    write_missing_modality_list(missing, out / 'test_missing_modality.csv')

    # Write split metadata
    meta = {
        'total_samples': len(samples),
        'train_samples': len(train),
        'test_samples': len(test),
        'train_participants': train_p,
        'test_participants': test_p,
        'missing_modality_samples': len(missing),
        'num_classes': len(ACTION_LABELS),
        'action_labels': ACTION_LABELS,
    }
    import json
    with open(out / 'split_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nFiles written to {out}/:")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")


if __name__ == '__main__':
    main()
