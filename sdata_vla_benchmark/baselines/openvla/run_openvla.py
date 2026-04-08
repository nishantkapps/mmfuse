#!/usr/bin/env python3
"""
Placeholder: wire OpenVLA frozen inference + manifest iteration.

Install OpenVLA deps separately, then implement forward + decode -> pred_idx.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    p = argparse.ArgumentParser(description="OpenVLA frozen eval (stub — implement with OpenVLA deps).")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(
        "Stub: install OpenVLA and implement inference. See README.md in this folder.",
        file=sys.stderr,
    )
    out = {
        "model": "openvla",
        "checkpoint": "not_run",
        "metrics": {"accuracy": 0.0, "macro_f1": 0.0, "n": 0},
        "predictions": [],
        "error": "Implement run_openvla.py with the OpenVLA library",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
