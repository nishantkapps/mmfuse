#!/usr/bin/env python3
"""
Export train/test manifest matching training/train_sdata_attention.py:
- Stratified 90/10 split on unique (cam1, cam2) pairs, random_state=42
- Train: all augmentation indices v for train pairs
- Test: only v==0 for test pairs

Sample enumeration is duplicated from SDataDataset.__init__ (no mmfuse import) so this
script runs with only stdlib + scikit-learn. Keep in sync with training.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sklearn.model_selection import train_test_split


def _get_audio_path(pdir: Path):
    m1 = list(pdir.glob("*_m1_*.wav"))
    return m1[0] if m1 else None


def _enumerate_samples(
    root_dir: Path,
    cross_pair: bool = False,
    augment_variations: int = 16,
) -> list[tuple]:
    """Returns the same list as SDataDataset.samples: (audio_path, cam1, cam2, label, v)."""
    augment_variations = max(1, augment_variations)
    samples = []
    part_dirs = sorted(d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("part"))
    part_to_label = {p.name: i for i, p in enumerate(part_dirs)}

    for part_dir in part_dirs:
        label = part_to_label[part_dir.name]
        video_pairs = []
        audio_paths = []
        for pdir in part_dir.iterdir():
            if not pdir.is_dir():
                continue
            cam1 = list(pdir.glob("*_c1_*.mp4"))
            cam2 = list(pdir.glob("*_c2_*.mp4"))
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
                audio_path = (
                    audio_paths[i // augment_variations]
                    if i // augment_variations < len(audio_paths)
                    else None
                )
                samples.append((audio_path, cam1, cam2, label, v))
    return samples


def build_rows(
    dataset_root: Path,
    cross_pair: bool = False,
    augment_variations: int = 16,
):
    samples = _enumerate_samples(dataset_root, cross_pair=cross_pair, augment_variations=augment_variations)
    pair_to_label = {}
    for audio_path, cam1, cam2, label, v in samples:
        key = (str(cam1), str(cam2))
        pair_to_label[key] = label
    unique_pairs = list(pair_to_label.keys())
    pair_labels = [pair_to_label[p] for p in unique_pairs]
    train_pairs, test_pairs = train_test_split(
        unique_pairs, test_size=0.1, stratify=pair_labels, random_state=42
    )
    train_set = set(train_pairs)
    test_set = set(test_pairs)

    rows = []
    for i, (audio_path, cam1, cam2, label, v) in enumerate(samples):
        key = (str(cam1), str(cam2))
        if key in train_set:
            split = "train"
        elif key in test_set and v == 0:
            split = "test"
        else:
            continue
        ap = str(audio_path) if audio_path else ""
        rows.append(
            {
                "sample_id": i,
                "split": split,
                "label": label,
                "cam1": str(cam1),
                "cam2": str(cam2),
                "audio": ap,
                "aug_v": v,
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser(description="Build SData train/test manifest (matches train_sdata_attention split).")
    p.add_argument("--dataset", type=Path, required=True, help="Path to sdata root (e.g. dataset/sdata)")
    p.add_argument("--out", type=Path, default=Path("sdata_vla_benchmark/manifests/sdata_manifest.csv"))
    p.add_argument("--cross-pair", action="store_true", help="Match training --cross-pair (cross_pair_audio_video)")
    p.add_argument("--augment-variations", type=int, default=16)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.dataset, cross_pair=args.cross_pair, augment_variations=args.augment_variations)
    fields = ["sample_id", "split", "label", "cam1", "cam2", "audio", "aug_v"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_test = sum(1 for r in rows if r["split"] == "test")
    print(f"Wrote {args.out} | train={n_train} test={n_test} total={len(rows)}")


if __name__ == "__main__":
    main()
