#!/usr/bin/env python3
"""
SayCan (Google Research) — language-grounded affordance + low-level policies.

There is **no** single Hugging Face checkpoint that reproduces the full SayCan stack (LLM scoring,
ViLD / affordance models, PyBullet policies) as an 8-way discrete classifier on arbitrary RGB clips.

This module records a **benchmark-aligned JSON** (same manifest / split) with `metrics: null` so the
orchestrator and `aggregate_results.py` can list SayCan alongside other rows. For numbers, you must
integrate the official pipeline (see note) or cite SayCan as related work only.

References: https://say-can.github.io/ , https://github.com/google-research/google-research/tree/master/saycan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdata_vla_benchmark.frozen.common import filter_split, load_manifest_rows, write_json


def run_saycan(manifest: Path, split: str, output: Path) -> dict:
    rows = filter_split(load_manifest_rows(manifest), split)
    setup_doc = REPO_ROOT / "sdata_vla_benchmark" / "EXTERNAL_BASELINES.md"
    req_sc = REPO_ROOT / "sdata_vla_benchmark" / "requirements-saycan.txt"
    out = {
        "model": "saycan",
        "checkpoint": "n/a (multi-component system; not one HF id)",
        "frozen": True,
        "split": split,
        "manifest": str(manifest),
        "metrics": None,
        "predictions": [],
        "n_manifest_rows": len(rows),
        "setup_doc": str(setup_doc),
        "requirements_file": str(req_sc),
        "note": (
            "Prediction stack: OpenAI API (GPT-style), OpenAI CLIP, Flax/Jax policies, PyBullet, TensorFlow/ViLD, "
            "gdown assets — see google-research/saycan SayCan-Robot-Pick-Place.ipynb and requirements-saycan.txt. "
            f"Not one checkpoint on SData RGB; details: {setup_doc}"
        ),
    }
    write_json(output, out)
    return out


def main():
    p = argparse.ArgumentParser(description="SayCan placeholder JSON (no single-model inference).")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="test")
    args = p.parse_args()
    run_saycan(args.manifest, args.split, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
