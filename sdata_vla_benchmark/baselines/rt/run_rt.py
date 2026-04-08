#!/usr/bin/env python3
"""Stub for RT-1/RT-2 style frozen policies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print("Stub: implement RT policy inference. See README.md", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "model": "rt",
                "metrics": {"accuracy": 0.0, "macro_f1": 0.0, "n": 0},
                "predictions": [],
                "error": "Implement run_rt.py",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
