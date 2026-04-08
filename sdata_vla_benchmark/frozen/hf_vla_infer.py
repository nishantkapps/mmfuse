#!/usr/bin/env python3
"""
Run **any** frozen Hugging Face VLA that exposes AutoModelForVision2Seq + generate(image, text).

Use this when you want a baseline other than OpenVLA (same eval protocol as openvla_infer.py).

Examples:
  python hf_vla_infer.py --model-id openvla/openvla-7b-finetuned-libero-spatial --report-as openvla_libero_spatial ...
  python hf_vla_infer.py --model-id <org>/<your-vla> --report-as my_vla ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdata_vla_benchmark.frozen.hf_vla_core import default_report_name, run_hf_vision2seq_vla


def main():
    p = argparse.ArgumentParser(description="Generic frozen HF Vision2Seq VLA eval on SData manifest.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--model-id", required=True, help="HF repo id (must support Vision2Seq + generate)")
    p.add_argument(
        "--report-as",
        default=None,
        help="Label in JSON (default: slug from model-id)",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--cam", choices=("cam1", "cam2"), default="cam1")
    args = p.parse_args()

    lbl = args.report_as or default_report_name(args.model_id)
    run_hf_vision2seq_vla(
        args.manifest,
        args.split,
        args.output,
        args.model_id,
        args.device,
        args.cam,
        model_label=lbl,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
