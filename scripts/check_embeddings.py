#!/usr/bin/env python3
"""
Check precomputed embeddings in an embeddings dir (from precompute_video_text.py).
Prints config, sample shapes, and stats (min/max/mean, NaN, all-zero) so you can
verify embeddings were created correctly.

Usage:
  python scripts/check_embeddings.py --embeddings-dir embeddings/vima_bench
  python scripts/check_embeddings.py --embeddings-dir embeddings/nextqa --samples 5
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description="Inspect precomputed embedding .pt files")
    p.add_argument("--embeddings-dir", type=Path, required=True, help="Path to embeddings dir (has config.json and *.pt)")
    p.add_argument("--samples", type=int, default=3, help="Number of .pt files to open (first, spread, last)")
    args = p.parse_args()

    emb_dir = args.embeddings_dir.resolve()
    if not emb_dir.is_dir():
        print(f"Error: not a directory: {emb_dir}")
        return 1

    config_path = emb_dir / "config.json"
    if not config_path.exists():
        print(f"Error: no config.json in {emb_dir}")
        return 1
    with open(config_path) as f:
        config = json.load(f)
    print("Config:", json.dumps(config, indent=2))

    pt_files = sorted(emb_dir.glob("*.pt"))
    if not pt_files:
        print(f"Error: no .pt files in {emb_dir}")
        return 1
    print(f"\nTotal .pt files: {len(pt_files)}")

    vision_dim = config.get("vision_dim")
    expected_keys = {"vision_camera1", "vision_camera2", "audio", "text", "target"}

    n = min(args.samples, len(pt_files))
    indices = [0] if n == 1 else [0, len(pt_files) // 2, len(pt_files) - 1][:n]
    if n > 3:
        step = max(1, (len(pt_files) - 1) // (n - 1))
        indices = [min(i * step, len(pt_files) - 1) for i in range(n)]

    for idx in indices:
        path = pt_files[idx]
        data = torch.load(path, map_location="cpu", weights_only=True)
        keys = set(data.keys())
        print(f"\n--- {path.name} ---")
        print(f"  Keys: {sorted(keys)}")
        missing = expected_keys - keys
        if missing:
            print(f"  WARNING: missing keys: {missing}")
        for k in sorted(keys):
            v = data[k]
            if isinstance(v, torch.Tensor):
                shp = tuple(v.shape)
                finite = torch.isfinite(v)
                nnan = (~finite).sum().item()
                nzero = (v == 0).sum().item()
                total = v.numel()
                all_zero = nzero == total
                if v.is_floating_point():
                    stats = f"min={v.min().item():.6f} max={v.max().item():.6f} mean={v.float().mean().item():.6f}"
                else:
                    stats = f"min={v.min().item()} max={v.max().item()}"
                print(f"  {k}: shape={shp} {stats} nan_count={nnan} all_zero={all_zero}")
                if vision_dim and k in ("vision_camera1", "vision_camera2") and shp != (vision_dim,):
                    print(f"    WARNING: expected shape ({vision_dim},) for vision_dim in config")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")

    # Optional: quick scan for any file with all-zero vision or NaN
    print("\n--- Quick scan (first 20 files) ---")
    nan_vision = 0
    zero_vision = 0
    for path in pt_files[:20]:
        data = torch.load(path, map_location="cpu", weights_only=True)
        for k in ("vision_camera1", "vision_camera2"):
            if k in data and isinstance(data[k], torch.Tensor):
                t = data[k]
                if not torch.isfinite(t).all():
                    nan_vision += 1
                    break
                if (t == 0).all():
                    zero_vision += 1
                    break
    if nan_vision or zero_vision:
        print(f"  In first 20 files: {nan_vision} with NaN in vision, {zero_vision} with all-zero vision")
    else:
        print("  First 20 files: no NaN or all-zero vision in vision_camera1/2")
    return 0


if __name__ == "__main__":
    exit(main())
