#!/usr/bin/env python3
"""
CLI: run the default synthetic experiment, save JSON/CSV/PNGs.

Usage (from repository root):

  python -m robot_policy_eval.run_evaluation --output-dir robot_policy_eval/outputs/run1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_policy_eval.config import ExperimentConfig
from robot_policy_eval.experiments.pipeline import default_policies, run_default_experiment, save_outputs
from robot_policy_eval.visualization.plots import (
    plot_3d_three_trajectories,
    plot_3d_trajectories,
    plot_error_vs_time,
    plot_error_vs_timestep_three_models,
    plot_occlusion_curve,
    plot_robustness_curve,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Robot policy evaluation (physio vs VLA vs hybrid).")
    ap.add_argument("--output-dir", type=Path, default=Path("robot_policy_eval/outputs/latest"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-subjects", type=int, default=10)
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib figures (JSON/CSV only).",
    )
    args = ap.parse_args()

    cfg = ExperimentConfig(random_seed=args.seed, output_dir=str(args.output_dir))

    suite, train_ds, test_ds = run_default_experiment(
        cfg=cfg,
        n_subjects=args.n_subjects,
    )
    save_outputs(suite, args.output_dir, cfg)

    if not args.no_plots and len(test_ds) > 0:
        rng = __import__("numpy").random.default_rng(cfg.random_seed)
        policies = default_policies(rng)
        gt0 = test_ds[0]
        obs = {"ground_truth": gt0}
        preds_by_name: dict[str, "Trajectory"] = {}
        for pol in policies:
            preds_by_name[pol.name] = pol.predict(obs)

        # Paper-style combined figures (A) + (B)
        physio = preds_by_name.get("dummy_physio")
        vla = preds_by_name.get("dummy_vla")
        hybrid = preds_by_name.get("hybrid_vla_physio")
        if physio is not None and vla is not None:
            plot_3d_three_trajectories(
                gt0,
                physio,
                vla,
                args.output_dir / "fig_paper_trajectory_gt_physio_vla.png",
                hybrid=hybrid,
                title="Trajectory comparison: GT vs Physio vs VLA",
            )
        label_map = {
            "dummy_physio": "Physio (ours)",
            "dummy_vla": "VLA",
            "hybrid_vla_physio": "Hybrid",
        }
        preds_labeled = {label_map.get(k, k): v for k, v in preds_by_name.items()}
        plot_error_vs_timestep_three_models(
            gt0,
            preds_labeled,
            args.output_dir / "fig_paper_error_vs_timestep.png",
        )

        for pol in policies:
            pred = preds_by_name[pol.name]
            plot_3d_trajectories(
                gt0,
                pred,
                args.output_dir / f"fig_3d_{pol.name}.png",
                title=f"{pol.name}: GT vs pred",
            )
            plot_error_vs_time(gt0, pred, args.output_dir / f"fig_err_time_{pol.name}.png")

        plot_robustness_curve(suite, args.output_dir / "fig_robustness_noise.png")
        plot_occlusion_curve(suite, args.output_dir / "fig_robustness_occlusion.png")

    summary_path = args.output_dir / "summary.json"
    split = suite.get("split", {})
    with open(summary_path, "w") as f:
        json.dump(
            {
                "output_dir": str(args.output_dir),
                "split": split,
                "aggregate_keys": list(suite.get("aggregates", {}).keys()),
            },
            f,
            indent=2,
        )
    print(f"Wrote metrics and tables under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
