#!/usr/bin/env python3
"""
Trim audio to match video duration.

Why the discrepancy happens:
- Video: one frame written per loop iteration. With dual cameras + imshow, the loop
  often runs slower than real-time (e.g. 15 fps instead of 30). So in 5 min wall-clock
  you get fewer frames → shorter video when played at 30 fps.
- Audio: records in real-time for the full wall-clock duration of the capture.
- Result: e.g. 5 min audio, 2.5 min video (loop ran at ~half speed).

This script trims the audio to the video duration, keeping the first N seconds.
No resampling—same sample rate, same content, just truncated.
"""
import argparse
import wave
from pathlib import Path

import cv2
import numpy as np


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds (frame_count / fps)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if frame_count <= 0:
        raise ValueError(f"Invalid frame count for {video_path}")
    return frame_count / fps


def trim_audio_to_duration(
    wav_in: str,
    wav_out: str,
    duration_sec: float,
    sr: int = 16000,
    channels: int = 1
) -> None:
    """
    Trim WAV to first `duration_sec` seconds. Writes to wav_out.
    Same sample rate, no resampling.
    """
    with wave.open(wav_in, 'rb') as wf:
        nchan = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        file_sr = wf.getframerate()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)

    total_sec = nframes / file_sr
    if duration_sec >= total_sec:
        # Nothing to trim; copy
        import shutil
        shutil.copy(wav_in, wav_out)
        return

    # Wave "frame" = one sample per channel; for mono, frames = samples
    keep_frames = int(duration_sec * file_sr)
    bytes_per_frame = sampwidth * nchan
    keep_bytes = keep_frames * bytes_per_frame
    trimmed = frames[:keep_bytes]

    with wave.open(wav_out, 'wb') as wf:
        wf.setnchannels(nchan)
        wf.setsampwidth(sampwidth)
        wf.setframerate(file_sr)
        wf.writeframes(trimmed)


def process_trial_folder(trial_dir: Path, dry_run: bool, suffix: str = None) -> int:
    """Process one trial folder: trim mic.wav and mic_speech.wav to match cam1.mp4."""
    cam1 = trial_dir / 'cam1.mp4'
    if not cam1.exists():
        print(f"  Skip (no cam1.mp4): {trial_dir}")
        return 0
    try:
        duration_sec = get_video_duration(str(cam1))
    except Exception as e:
        print(f"  Error reading video: {trial_dir} - {e}")
        return 1

    count = 0
    for wav_name in ['mic.wav', 'mic_speech.wav']:
        wav_path = trial_dir / wav_name
        if not wav_path.exists():
            continue
        with wave.open(str(wav_path), 'rb') as wf:
            audio_sec = wf.getnframes() / wf.getframerate()
        if audio_sec <= duration_sec:
            continue
        out_path = wav_path
        if suffix:
            out_path = wav_path.parent / f"{wav_path.stem}{suffix}.wav"
        if dry_run:
            print(f"  Would trim {wav_name} -> {out_path.name}: {audio_sec:.1f}s -> {duration_sec:.1f}s")
        else:
            trim_audio_to_duration(str(wav_path), str(out_path), duration_sec)
            print(f"  Trimmed {wav_name} -> {out_path.name}: {audio_sec:.1f}s -> {duration_sec:.1f}s")
        count += 1
    return count


def main():
    p = argparse.ArgumentParser(description="Trim audio to match video duration")
    p.add_argument('--video', help='Reference video (cam1.mp4 or cam2.mp4)')
    p.add_argument('--audio', help='Audio file to trim (mic.wav or mic_speech.wav)')
    p.add_argument('--out', default=None, help='Output path (default: overwrite --audio)')
    p.add_argument('--trial-dir', help='Trial folder (trim mic.wav, mic_speech.wav to match cam1.mp4)')
    p.add_argument('--batch', help='Root dir: find all trial folders (with meta.json) and trim')
    p.add_argument('--suffix', default=None, help='Save trimmed output as {name}{suffix}.wav (e.g. _trimmed -> mic_trimmed.wav). Preserves originals.')
    p.add_argument('--dry-run', action='store_true', help='Print only, do not write')
    args = p.parse_args()

    if args.batch:
        root = Path(args.batch)
        trials = [p.parent for p in root.rglob('meta.json')]
        print(f"Found {len(trials)} trial folders")
        for td in trials:
            process_trial_folder(td, args.dry_run, args.suffix)
        return 0

    if args.trial_dir:
        process_trial_folder(Path(args.trial_dir), args.dry_run, args.suffix)
        return 0

    if not args.video or not args.audio:
        p.print_help()
        print("\nExamples:")
        print("  # Single file (use --out to save elsewhere):")
        print("  python trim_audio_to_video.py --video trial/cam1.mp4 --audio trial/mic.wav --out trial/mic_trimmed.wav")
        print("  # Trial folder (preserve originals, save as mic_trimmed.wav):")
        print("  python trim_audio_to_video.py --trial-dir dataset/V01/.../T0001 --suffix _trimmed")
        print("  # Batch all trials, preserve originals:")
        print("  python trim_audio_to_video.py --batch dataset/ --suffix _trimmed")
        return 1

    video_path = Path(args.video)
    audio_path = Path(args.audio)
    out_path = Path(args.out) if args.out else audio_path

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return 1
    if not audio_path.exists():
        print(f"Error: Audio not found: {audio_path}")
        return 1

    duration_sec = get_video_duration(video_path)
    with wave.open(str(audio_path), 'rb') as wf:
        audio_frames = wf.getnframes()
        audio_sr = wf.getframerate()
        audio_sec = audio_frames / audio_sr

    print(f"Video duration: {duration_sec:.2f}s")
    print(f"Audio duration: {audio_sec:.2f}s")
    print(f"Trim audio to:  {duration_sec:.2f}s")

    if args.dry_run:
        return 0

    trim_audio_to_duration(
        str(audio_path),
        str(out_path),
        duration_sec,
        sr=audio_sr
    )
    print(f"Wrote trimmed audio to {out_path}")
    return 0


if __name__ == '__main__':
    exit(main())
