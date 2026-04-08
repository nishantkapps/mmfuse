"""Run MMFuse + baseline policies on trajectory JSON (SData export or synthetic)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_policy_eval.baselines.factory import build_policies_by_name, default_sota_names
from robot_policy_eval.config import ExperimentConfig
from robot_policy_eval.cross_model.run import evaluate_policies_on_trajectory_dataset
from robot_policy_eval.experiments.pipeline import save_outputs


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Compare MMFuse and frozen-style baselines in one trajectory metric suite. "
            "Requires trajectory JSON with meta.cam1 (+ optional cam2, audio) from build_sdata_trajectories."
        ),
    )
    p.add_argument("--dataset", type=Path, required=True, help="trajectories.json")
    p.add_argument("--repo-root", type=Path, default=Path.cwd(), help="mmfuse repo root (paths + scripts/)")
    p.add_argument(
        "--movement-config",
        type=Path,
        default=None,
        help="sdata_movement_config.yaml (default: <repo-root>/config/sdata_movement_config.yaml)",
    )
    p.add_argument("--mmfuse-checkpoint", type=Path, default=None, help="Required if baselines include mmfuse")
    p.add_argument("--mmfuse-mode", choices=("full", "vision_only"), default="full")
    p.add_argument(
        "--baselines",
        nargs="+",
        default=default_sota_names(),
        help=f"Policy keys (default: {default_sota_names()})",
    )
    p.add_argument("--output-dir", type=Path, default=Path("robot_policy_eval/outputs/sota"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda / cpu for torch models")
    p.add_argument("--test-fraction", type=float, default=None)
    p.add_argument("--test-subjects", nargs="*", default=None)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    movement_cfg = args.movement_config
    if movement_cfg is None:
        movement_cfg = repo_root / "config" / "sdata_movement_config.yaml"
    if not movement_cfg.is_file():
        print(f"Movement config not found: {movement_cfg}", file=sys.stderr)
        sys.exit(1)

    try:
        policies, columns = build_policies_by_name(
            list(args.baselines),
            repo_root=repo_root,
            movement_yaml=movement_cfg,
            mmfuse_checkpoint=args.mmfuse_checkpoint,
            mmfuse_mode=args.mmfuse_mode,
            device=args.device,
            seed=args.seed,
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    cfg = ExperimentConfig(random_seed=args.seed, output_dir=str(args.output_dir))
    suite, _, test_ds = evaluate_policies_on_trajectory_dataset(
        policies,
        args.dataset,
        cfg,
        test_subject_ids=args.test_subjects,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    suite["policy_columns"] = columns
    suite["comparison"] = {
        "kind": "trajectory_capability",
        "movement_config": str(movement_cfg.resolve()),
        "baselines": list(args.baselines),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_outputs(suite, args.output_dir, cfg)

    if not args.no_plots and len(test_ds) > 0:
        from robot_policy_eval.cross_model.run import _write_optional_plots

        _write_optional_plots(suite, policies, test_ds, args.output_dir, args.seed)

    print(f"Wrote metrics and tables under {args.output_dir.resolve()}")
    print(json.dumps({"policies": [c[0] for c in columns], "display": [c[1] for c in columns]}, indent=2))


if __name__ == "__main__":
    main()
