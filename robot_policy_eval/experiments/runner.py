"""
Experiment runner: normal / noisy / subject-split evaluation and aggregation.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from robot_policy_eval.config import ExperimentConfig
from robot_policy_eval.data.dataset import TrajectoryDataset
from robot_policy_eval.data.types import Trajectory
from robot_policy_eval.metrics.force_metrics import mean_absolute_force_error
from robot_policy_eval.metrics.smoothness import mean_jerk
from robot_policy_eval.metrics.success_safety import compute_safety_violations, task_success
from robot_policy_eval.metrics.trajectory_error import (
    dtw_on_trajectories,
    mean_l2_trajectory_error,
)
from robot_policy_eval.policies.base import Policy
from robot_policy_eval.robustness.perturbations import (
    add_gaussian_position_noise,
    random_timestep_occlusion,
)


def _evaluate_one(
    policy: Policy,
    gt: Trajectory,
    observation: dict[str, Any],
    cfg: ExperimentConfig,
) -> dict[str, Any]:
    pred = policy.predict(observation)
    n_res = cfg.metric_resample_points
    return {
        "mean_l2_position_error": mean_l2_trajectory_error(pred, gt, n_resample=n_res),
        "dtw_position": dtw_on_trajectories(pred, gt),
        "mean_jerk_pred": mean_jerk(pred),
        "mean_abs_force_error": mean_absolute_force_error(pred, gt, n_resample=n_res),
        "safety_violations_pred": compute_safety_violations(pred, cfg.force_safe_max),
        "success": task_success(
            pred,
            gt,
            trajectory_rmse_threshold=cfg.trajectory_success_rmse_threshold,
            force_safe_max=cfg.force_safe_max,
            n_resample=n_res,
        ),
        "n_pred_steps": len(pred),
        "n_gt_steps": len(gt),
    }


def run_evaluation_suite(
    policies: list[Policy],
    test_dataset: TrajectoryDataset,
    cfg: ExperimentConfig | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Runs:
      * normal (clean GT in observation)
      * Gaussian position noise on input observations (multiple sigmas)
      * random occlusion on input observations (multiple drop probs)

    Metrics aggregated by mean over episodes per (policy, condition).
    """
    cfg = cfg or ExperimentConfig()
    rng = rng or np.random.default_rng(cfg.random_seed)

    results: dict[str, Any] = {
        "config": {k: v for k, v in asdict(cfg).items() if k != "extra"},
        "per_episode": [],
        "aggregates": {},
    }

    # Condition grid: (name, builder(obs_gt: Trajectory) -> dict observation)
    def _obs_normal(tr: Trajectory) -> dict[str, Any]:
        return {"ground_truth": tr}

    conditions: list[tuple[str, Any]] = [("normal", _obs_normal)]

    for sig in cfg.position_noise_sigmas:
        if sig <= 0:
            continue

        def _make_noise(sigma: float):
            def _fn(tr: Trajectory) -> dict[str, Any]:
                return {
                    "ground_truth": add_gaussian_position_noise(tr, sigma, rng),
                    "noise_sigma": sigma,
                }

            return _fn

        conditions.append((f"pos_noise_{sig}", _make_noise(sig)))

    for p_occ in cfg.occlusion_drop_probs:
        if p_occ <= 0:
            continue

        def _make_occ(p: float):
            def _fn(tr: Trajectory) -> dict[str, Any]:
                return {
                    "ground_truth": random_timestep_occlusion(tr, p, rng),
                    "occlusion_p": p,
                }

            return _fn

        conditions.append((f"occlusion_{p_occ}", _make_occ(p_occ)))

    agg: dict[str, dict[str, list[float]]] = {}

    for pol in policies:
        agg[pol.name] = {}
        for cond_name, obs_builder in conditions:
            metrics_lists: dict[str, list] = {
                "mean_l2_position_error": [],
                "dtw_position": [],
                "mean_jerk_pred": [],
                "mean_abs_force_error": [],
                "safety_violations_pred": [],
                "success": [],
            }
            for ep in range(len(test_dataset)):
                gt = test_dataset[ep]
                obs = obs_builder(gt)
                m = _evaluate_one(pol, gt, obs, cfg)
                m["policy"] = pol.name
                m["condition"] = cond_name
                m["episode_id"] = gt.episode_id
                m["subject_id"] = gt.subject_id
                results["per_episode"].append(m)
                for k in metrics_lists:
                    metrics_lists[k].append(m[k])
            key = cond_name
            agg[pol.name][key] = {
                k: float(np.mean(v)) if k != "success" else float(np.mean([float(x) for x in v]))
                for k, v in metrics_lists.items()
            }
            agg[pol.name][key]["success_rate"] = agg[pol.name][key]["success"]

    results["aggregates"] = agg
    return results


def comparison_table_rows(
    suite_result: dict[str, Any],
    condition: str = "normal",
) -> list[dict[str, Any]]:
    """Flatten aggregates for CSV export for one condition."""
    rows = []
    ag = suite_result.get("aggregates", {})
    for policy_name, conds in ag.items():
        if condition not in conds:
            continue
        row = {"policy": policy_name, "condition": condition, **conds[condition]}
        rows.append(row)
    return rows
