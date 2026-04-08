"""Task success (thresholded trajectory error + safety) and force violation counts."""
from __future__ import annotations

import numpy as np

from robot_policy_eval.data.types import Trajectory
from robot_policy_eval.metrics.trajectory_error import mean_l2_trajectory_error


def compute_safety_violations(traj: Trajectory, force_safe_max: float) -> int:
    """Count timesteps where |force| > threshold (strictly greater)."""
    return int(np.sum(traj.forces > force_safe_max))


def task_success(
    pred: Trajectory,
    gt: Trajectory,
    *,
    trajectory_rmse_threshold: float,
    force_safe_max: float,
    n_resample: int = 64,
) -> bool:
    """
    Success iff mean L2 trajectory error < threshold AND predicted trajectory has
    no safety violations (force at any timestep above safe max).
    """
    err = mean_l2_trajectory_error(pred, gt, n_resample=n_resample)
    violations = compute_safety_violations(pred, force_safe_max)
    return bool(err < trajectory_rmse_threshold and violations == 0)
