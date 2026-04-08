"""
Trajectory alignment: mean L2 error after temporal resampling, and DTW distance.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from robot_policy_eval.data.types import Trajectory


def resample_trajectory_positions(
    traj: Trajectory,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample positions to `n_points` uniformly in *time* (uses trajectory timestamps).
    Returns (positions (n_points, 3), uniform_time_axis (n_points,)).
    """
    if len(traj) < 2:
        p = traj.positions
        if len(p) == 0:
            return np.zeros((n_points, 3)), np.linspace(0, 1, n_points)
        return np.repeat(p[:1], n_points, axis=0), np.linspace(
            float(traj.timestamps[0]), float(traj.timestamps[0]), n_points
        )

    t = traj.timestamps
    p = traj.positions
    t_min, t_max = float(t[0]), float(t[-1])
    if t_max <= t_min:
        t_max = t_min + 1e-6
    t_new = np.linspace(t_min, t_max, n_points)
    f = interp1d(t, p, axis=0, kind="linear", fill_value="extrapolate")
    return np.asarray(f(t_new), dtype=np.float64), t_new


def mean_l2_trajectory_error(
    pred: Trajectory,
    gt: Trajectory,
    n_resample: int = 64,
) -> float:
    """
    Mean Euclidean distance between predicted and GT positions after both are
    resampled to `n_resample` points on each's own time base (then compared timestep-wise).
    """
    pp, _ = resample_trajectory_positions(pred, n_resample)
    gp, _ = resample_trajectory_positions(gt, n_resample)
    return float(np.linalg.norm(pp - gp, axis=1).mean())


def per_timestep_position_errors(
    pred: Trajectory,
    gt: Trajectory,
    n_resample: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (time_axis, l2_errors) after common resampling."""
    pp, t = resample_trajectory_positions(pred, n_resample)
    gp, _ = resample_trajectory_positions(gt, n_resample)
    err = np.linalg.norm(pp - gp, axis=1)
    return t, err


def dtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
) -> float:
    """
    Classic DTW with Euclidean local cost. seq_* shape (T, D).

    O(T1 * T2) dynamic programming; suitable for moderate T (hundreds).
    """
    seq_a = np.asarray(seq_a, dtype=np.float64)
    seq_b = np.asarray(seq_b, dtype=np.float64)
    n, m = seq_a.shape[0], seq_b.shape[0]
    if n == 0 or m == 0:
        return 0.0

    inf = float("inf")
    dtw = np.full((n + 1, m + 1), inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(seq_a[i - 1] - seq_b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m] / max(n, m))  # normalized average warping cost


def dtw_on_trajectories(pred: Trajectory, gt: Trajectory) -> float:
    """DTW on 3D position sequences (raw lengths, no resampling)."""
    return dtw_distance(pred.positions, gt.positions)
