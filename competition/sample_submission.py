#!/usr/bin/env python3
"""
MIRA Challenge — Generate Sample Submission
Creates a random baseline submission CSV that participants can use as a format reference.

Usage:
  python competition/sample_submission.py \
      --test-samples competition/data/test_samples.csv \
      --output competition/data/sample_submission.csv
"""

import argparse
import csv
import random


ACTION_LABELS = ['Start', 'Go Here', 'Move Down', 'Move Up',
                 'Stop', 'Move Left', 'Move Right', 'Perfect']


def main():
    parser = argparse.ArgumentParser(description='Generate sample submission')
    parser.add_argument('--test-samples', required=True, help='Test samples CSV (no labels)')
    parser.add_argument('--output', required=True, help='Output submission CSV')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.test_samples) as f:
        reader = csv.DictReader(f)
        samples = [r['sample'] for r in reader]

    with open(args.output, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'action', 'dx', 'dy'])
        for s in samples:
            action = rng.choice(ACTION_LABELS)
            dx = round(rng.uniform(-5, 5), 2)
            dy = round(rng.uniform(-5, 5), 2)
            w.writerow([s, action, dx, dy])

    print(f"Wrote sample submission: {args.output} ({len(samples)} samples)")


if __name__ == '__main__':
    main()
