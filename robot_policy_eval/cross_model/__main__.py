"""CLI: evaluate built-in demo policies on a trajectory JSON dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_policy_eval.cross_model.run import run_from_cli


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Cross-model evaluation on YOUR trajectory dataset (JSON). "
            "Same metrics for every policy — see paper/CROSS_MODEL_EVAL.md."
        ),
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="JSON file: {trajectories: [...]} or a list of trajectory dicts",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("robot_policy_eval/outputs/cross_model"),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help="Fraction of subjects to hold out for test (default: last ~1/3 of subjects)",
    )
    ap.add_argument(
        "--test-subjects",
        nargs="*",
        default=None,
        help="Explicit test subject IDs (overrides --test-fraction)",
    )
    ap.add_argument(
        "--policies",
        nargs="+",
        default=["dummy_physio", "dummy_vla", "hybrid"],
        choices=["dummy_physio", "dummy_vla", "hybrid"],
        help="Built-in demo policies only; for real models use the Python API with your Policy classes",
    )
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument(
        "--print-split",
        action="store_true",
        help="Print train/test subject split to stdout",
    )
    args = ap.parse_args()

    suite = run_from_cli(
        args.dataset,
        args.output_dir,
        list(args.policies),
        test_subject_ids=args.test_subjects,
        test_fraction=args.test_fraction,
        seed=args.seed,
        no_plots=args.no_plots,
    )
    if args.print_split:
        sp = suite.get("split", {})
        print(
            json.dumps(
                {
                    "train_subjects": sp.get("train_subjects"),
                    "test_subjects": sp.get("test_subjects"),
                    "n_test_episodes": sp.get("n_test_episodes"),
                },
                indent=2,
            )
        )
    print(f"Wrote metrics and tables under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
