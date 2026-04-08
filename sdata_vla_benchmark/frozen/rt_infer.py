#!/usr/bin/env python3
"""
RT-1 / RT-2 (paper names): Google's public release is not a single Hugging Face `transformers` checkpoint.

We **download** the closest hub-hosted robotics transformer weights (**RDT**) and record metadata.
For metrics, either extend `rdt_infer.py` with the upstream `RoboticsDiffusionTransformer` runner
or cite RDT explicitly in the paper as the frozen HF baseline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdata_vla_benchmark.frozen.rdt_infer import run_rdt


def run_rt(manifest: Path, split: str, output: Path, variant: str) -> dict:
    return run_rdt(manifest, split, output, variant)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--variant", choices=("rt1", "rt2"), default="rt1")
    args = p.parse_args()
    run_rt(args.manifest, args.split, args.output, args.variant)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
