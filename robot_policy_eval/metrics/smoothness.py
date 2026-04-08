"""
Smoothness via jerk (third time derivative of position), discrete approximation.
"""
from __future__ import annotations

import numpy as np

from robot_policy_eval.data.types import Trajectory


def mean_jerk(traj: Trajectory) -> float:
    """
    Mean L2 norm of jerk vector (3D) over interior timesteps.
    j ≈ (p[t+3] - 3p[t+2] + 3p[t+1] - p[t]) / dt^3 for uniform spacing (central differences variant).

    Uses actual timestamp spacing: approximate derivatives with np.gradient on each axis.
    """
    p = traj.positions
    t = traj.timestamps
    if len(p) < 4:
        return 0.0

    # Velocity
    dt = np.diff(t)
    dt = np.where(dt > 1e-9, dt, 1e-9)
    # Per-axis gradient w.r.t. time using second-order accurate edges
    vx = np.gradient(p[:, 0], t, edge_order=2)
    vy = np.gradient(p[:, 1], t, edge_order=2)
    vz = np.gradient(p[:, 2], t, edge_order=2)
    v = np.stack([vx, vy, vz], axis=1)

    ax = np.gradient(v[:, 0], t, edge_order=2)
    ay = np.gradient(v[:, 1], t, edge_order=2)
    az = np.gradient(v[:, 2], t, edge_order=2)
    a = np.stack([ax, ay, az], axis=1)

    jx = np.gradient(a[:, 0], t, edge_order=2)
    jy = np.gradient(a[:, 1], t, edge_order=2)
    jz = np.gradient(a[:, 2], t, edge_order=2)
    jerk = np.stack([jx, jy, jz], axis=1)

    norms = np.linalg.norm(jerk, axis=1)
    return float(np.mean(norms))
