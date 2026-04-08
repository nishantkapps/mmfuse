#!/usr/bin/env python3
"""
Print what is needed for RDT (RT proxy) and SayCan full prediction, and optional quick checks.

Does not download large checkpoints. Run from mmfuse repo root:

  python sdata_vla_benchmark/scripts/check_external_baselines.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SDATA_BENCH = REPO / "sdata_vla_benchmark"


def main() -> None:
    print("=== RDT (rt1/rt2 in run_frozen_benchmarks) ===\n")
    print("Docs: ", SDATA_BENCH / "EXTERNAL_BASELINES.md")
    print("Deps pin list: ", SDATA_BENCH / "requirements-rdt.txt")
    print("\nInstall RDT in a *separate* conda env (see EXTERNAL_BASELINES.md).")
    rdt_root = os.environ.get("RDT_ROOT", "").strip()
    if rdt_root:
        p = Path(rdt_root).expanduser()
        ok = p.is_dir() and (p / "scripts" / "agilex_model.py").is_file()
        print(f"RDT_ROOT={rdt_root!r} -> {'OK (scripts/agilex_model.py found)' if ok else 'MISSING or incomplete'}")
    else:
        print("RDT_ROOT not set (optional; set to your RoboticsDiffusionTransformer clone).")

    print("\n=== SayCan ===\n")
    print("Deps list: ", SDATA_BENCH / "requirements-saycan.txt")
    print("SayCan is multi-component (OpenAI API, CLIP, Flax, PyBullet, TF, gdown assets).")
    print("No single HF checkpoint for 8-class SData prediction — see EXTERNAL_BASELINES.md.")

    print("\n=== Optional import probes (non-fatal) ===\n")
    for name in ("huggingface_hub", "torch"):
        try:
            __import__(name)
            print(f"  {name}: import OK")
        except ImportError as e:
            print(f"  {name}: not available ({e})")


if __name__ == "__main__":
    main()
