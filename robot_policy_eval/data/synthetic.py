"""
Synthetic trajectory generation for demos and unit-style runs (no hardcoded file paths).
"""
from __future__ import annotations

import math

import numpy as np

from robot_policy_eval.data.dataset import TrajectoryDataset
from robot_policy_eval.data.types import Timestep, Trajectory


def _smooth_curve(
    t: np.ndarray,
    subject_offset: np.ndarray,
    amplitude: float = 0.15,
    freq: float = 1.2,
) -> np.ndarray:
    """3D position curve in [0,1]^3-ish workspace."""
    x = 0.3 + amplitude * np.sin(freq * t) + subject_offset[0]
    y = 0.3 + amplitude * np.cos(freq * t * 0.9) + subject_offset[1]
    z = 0.2 + 0.1 * t + 0.05 * np.sin(2 * freq * t) + subject_offset[2]
    return np.stack([x, y, z], axis=1)


def generate_synthetic_dataset(
    n_subjects: int = 10,
    episodes_per_subject: int = 5,
    n_timesteps: int = 40,
    dt: float = 0.02,
    rng: np.random.Generator | None = None,
) -> TrajectoryDataset:
    """
    Each subject has a fixed spatial bias; episodes add small noise.
    Ground-truth forces stay below a nominal safe band with rare spikes (for safety metrics).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    trajectories: list[Trajectory] = []
    for sid in range(n_subjects):
        subject_id = f"S{sid:02d}"
        offset = rng.normal(0, 0.03, size=3)
        for ep in range(episodes_per_subject):
            t_axis = np.arange(n_timesteps, dtype=np.float64) * dt
            pos = _smooth_curve(t_axis, offset, amplitude=0.12 + 0.02 * sid / n_subjects)
            pos += rng.normal(0, 0.002, size=pos.shape)

            # Orientation: gentle variation
            orient = np.zeros((n_timesteps, 3))
            orient[:, 0] = 0.05 * np.sin(1.5 * t_axis)
            orient[:, 1] = 0.03 * np.cos(1.2 * t_axis)
            orient[:, 2] = 0.02 * np.sin(t_axis)

            # Force (N): mostly 10–30, occasional spike
            force = 15.0 + 8.0 * np.abs(np.sin(0.7 * t_axis))
            if rng.random() < 0.15:
                idx = rng.integers(0, n_timesteps)
                force[idx : idx + 3] = 55.0 + rng.uniform(0, 10)  # safety violation window

            steps = [
                Timestep(
                    position=pos[i],
                    orientation=orient[i],
                    force=float(force[i]),
                    timestamp=float(t_axis[i]),
                )
                for i in range(n_timesteps)
            ]
            trajectories.append(
                Trajectory(
                    timesteps=steps,
                    subject_id=subject_id,
                    episode_id=f"{subject_id}_ep{ep}",
                    meta={"synthetic": True, "dt": dt},
                )
            )

    return TrajectoryDataset(trajectories, name="synthetic")
