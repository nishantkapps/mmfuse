#!/usr/bin/env python3
"""
Frozen OpenVLA (default HF id) — thin wrapper around hf_vla_core.

For **other** HF VLAs, use the same script with `--model-id` and `--report-as`, or `hf_vla_infer.py`.
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
    p = argparse.ArgumentParser(
        description="Frozen HF Vision2Seq VLA (default: openvla/openvla-7b). Same metrics as other hf_vla_core runners."
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument(
        "--model-id",
        default=os.environ.get("OPEN_VLA_MODEL_ID", os.environ.get("HF_VLA_MODEL_ID", "openvla/openvla-7b")),
        help="Any Hugging Face model that supports AutoModelForVision2Seq + image+text generate",
    )
    p.add_argument(
        "--report-as",
        default=os.environ.get("HF_VLA_REPORT_AS", "").strip() or None,
        help="Name in JSON/table (default: short slug from model-id, or 'openvla' if id contains openvla)",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--cam", choices=("cam1", "cam2"), default="cam1")
    args = p.parse_args()

    mid = args.model_id
    if args.report_as:
        lbl = args.report_as
    elif "openvla" in mid.lower():
        lbl = "openvla"
    else:
        lbl = default_report_name(mid)

    run_hf_vision2seq_vla(
        args.manifest,
        args.split,
        args.output,
        mid,
        args.device,
        args.cam,
        model_label=lbl,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
