#!/usr/bin/env python3
"""
MIRA Challenge — Extract Displacement Ground Truth from Overhead Camera
Uses MediaPipe hand/pose tracking to extract gripper start/end positions
from camera 2 (overhead) videos, producing (dx, dy) displacement vectors.

Usage:
  python tools/extract_trajectories.py \
      --dataset /path/to/sdata \
      --sample-list tools/data/test_samples.csv \
      --output tools/data/test_ground_truth.csv

Requires: mediapipe>=0.10, opencv-python
"""

import argparse
import csv
import cv2
import numpy as np
from pathlib import Path

import mediapipe as mp

WRIST_INDEX = 0
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "hand_landmarker.task")

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


def _action_from_sample(s):
    """Derive action label from either the 'action' column or the sample/path name."""
    if 'action' in s and s['action']:
        return s['action']
    for key in ('c2', 'c1', 'sample'):
        if key in s:
            for part_key, label in ACTION_LABELS.items():
                if part_key in s[key]:
                    return label
    return 'Unknown'


def extract_hand_positions(video_path, max_frames=None):
    """Track hand landmark positions across video frames using MediaPipe HandLandmarker (tasks API).
    Returns list of (x, y) normalized coordinates per frame, or None for frames with no detection."""
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames and frame_count > max_frames:
        frame_count = max_frames

    positions = []
    with HandLandmarker.create_from_options(options) as landmarker:
        for idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(idx * 1000.0 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                wrist = result.hand_landmarks[0][WRIST_INDEX]
                positions.append((wrist.x, wrist.y))
            else:
                positions.append(None)

    cap.release()
    return positions


def compute_displacement(positions, frame_width_cm=30.0, frame_height_cm=20.0):
    """Compute (dx, dy) in cm from first and last valid detected positions.
    frame_width_cm / frame_height_cm: physical dimensions of the camera's field of view."""
    valid = [(x, y) for p in positions if p is not None for x, y in [p]]
    if len(valid) < 2:
        return 0.0, 0.0

    x0, y0 = valid[0]
    x1, y1 = valid[-1]
    dx = (x1 - x0) * frame_width_cm
    dy = (y1 - y0) * frame_height_cm
    return round(dx, 2), round(dy, 2)


def main():
    parser = argparse.ArgumentParser(description='MIRA — Extract Trajectory Ground Truth')
    parser.add_argument('--dataset', default='dataset/sdata', help='Root sdata directory')
    parser.add_argument('--sample-list', required=True,
                        help='CSV with at least sample,c1,c2 columns (action derived from path if missing)')
    parser.add_argument('--output', required=True, help='Output ground truth CSV')
    parser.add_argument('--camera', choices=['c1', 'c2'], default='c2',
                        help='Which camera to track (default: c2 overhead)')
    parser.add_argument('--frame-width-cm', type=float, default=30.0,
                        help='Physical width of camera FOV in cm')
    parser.add_argument('--frame-height-cm', type=float, default=20.0,
                        help='Physical height of camera FOV in cm')
    args = parser.parse_args()

    root = Path(args.dataset)
    cam_col = args.camera

    with open(args.sample_list) as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    print(f"Extracting trajectories from {len(samples)} samples using {cam_col} camera...")

    results = []
    for i, s in enumerate(samples):
        video_path = root / s[cam_col]
        action = _action_from_sample(s)
        if not video_path.exists():
            print(f"  WARN: {video_path} not found, using (0, 0)")
            results.append({'sample': s['sample'], 'action': action, 'dx': 0.0, 'dy': 0.0})
            continue

        positions = extract_hand_positions(str(video_path))
        dx, dy = compute_displacement(positions, args.frame_width_cm, args.frame_height_cm)
        results.append({'sample': s['sample'], 'action': action, 'dx': dx, 'dy': dy})

        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            print(f"  Processed {i+1}/{len(samples)}")

    with open(args.output, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample', 'action', 'dx', 'dy'])
        for r in results:
            w.writerow([r['sample'], r['action'], f"{r['dx']:.2f}", f"{r['dy']:.2f}"])

    print(f"Wrote {args.output} ({len(results)} samples)")


if __name__ == '__main__':
    main()
