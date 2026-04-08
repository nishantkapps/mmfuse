"""
Input perturbations for robustness sweeps (applied to observations, not evaluation GT).
"""
from __future__ import annotations

import numpy as np

from robot_policy_eval.data.types import Timestep, Trajectory


def add_gaussian_position_noise(
    traj: Trajectory,
    sigma: float,
    rng: np.random.Generator,
) -> Trajectory:
    """Independent Gaussian noise on each (x,y,z)."""
    steps = []
    for ts in traj.timesteps:
        p = ts.position + rng.normal(0, sigma, 3)
        steps.append(
            Timestep(
                position=p,
                orientation=ts.orientation.copy(),
                force=ts.force,
                timestamp=ts.timestamp,
            )
        )
    return Trajectory(
        timesteps=steps,
        subject_id=traj.subject_id,
        episode_id=traj.episode_id + "_posnoise",
        meta={**traj.meta, "position_noise_sigma": sigma},
    )


def random_timestep_occlusion(
    traj: Trajectory,
    drop_probability: float,
    rng: np.random.Generator,
) -> Trajectory:
    """
    Randomly drops timesteps (simulates occlusion / frame loss). Keeps first and last.
    """
    if len(traj) <= 2 or drop_probability <= 0:
        return traj
    keep = [0]
    for i in range(1, len(traj) - 1):
        if rng.random() > drop_probability:
            keep.append(i)
    keep.append(len(traj) - 1)
    keep = sorted(set(keep))
    steps = [traj.timesteps[i] for i in keep]
    return Trajectory(
        timesteps=steps,
        subject_id=traj.subject_id,
        episode_id=traj.episode_id + "_occ",
        meta={**traj.meta, "occlusion_drop_p": drop_probability},
    )
