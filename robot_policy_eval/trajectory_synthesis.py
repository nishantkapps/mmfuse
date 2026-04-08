"""
Canonical EE trajectory synthesis from movement YAML deltas (shared GT + model predictions).

Used by SData export and by policies that output (delta_along, delta_lateral, magnitude).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_policy_eval.data.types import Timestep, Trajectory


def load_movements_yaml(path: Path | str) -> list[dict]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    movements = cfg.get("movements", [])
    if not movements:
        raise ValueError(f"No movements[] in {path}")
    return movements


def movement_triple_for_label(movements: list[dict], label: int) -> tuple[float, float, float]:
    if label < 0 or label >= len(movements):
        raise ValueError(f"label {label} out of range [0, {len(movements)})")
    m = movements[label]
    return (
        float(m.get("delta_along", 0)),
        float(m.get("delta_lateral", 0)),
        float(m.get("magnitude", 0)),
    )


def trajectory_from_delta_triple(
    delta_along: float,
    delta_lateral: float,
    magnitude: float,
    *,
    n_timesteps: int = 48,
    dt: float = 0.02,
    aug_v: int = 0,
    sample_id: int = 0,
    subject_id: str = "",
    episode_id: str = "",
    meta: dict[str, Any] | None = None,
) -> Trajectory:
    """Smooth EE path from continuous deltas (same construction as SData export)."""
    v = np.array([delta_along, delta_lateral, 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        direction = np.zeros(3, dtype=np.float64)
    else:
        direction = v / norm
    mag = float(magnitude)
    scale = 0.14 * max(mag, 0.08)
    t_idx = np.arange(n_timesteps, dtype=np.float64)
    alpha = 0.5 * (1.0 - np.cos(np.pi * t_idx / max(n_timesteps - 1, 1)))
    p0 = np.array([0.42, 0.40, 0.24], dtype=np.float64)
    jitter = 1e-3 * (aug_v % 17) + 1e-4 * (sample_id % 97)
    orth = np.array([-direction[1], direction[0], 0.0], dtype=np.float64)
    if np.linalg.norm(orth) > 1e-9:
        orth = orth / np.linalg.norm(orth)
    positions = (
        p0[None, :]
        + (direction * scale)[None, :] * alpha[:, None]
        + orth[None, :] * jitter * alpha[:, None]
    )
    positions[:, 2] += 0.015 * np.sin(2.0 * np.pi * t_idx / n_timesteps)
    orientations = np.zeros((n_timesteps, 3), dtype=np.float64)
    base_f = 10.0 + 5.0 * mag
    forces = base_f + 3.0 * np.sin(np.linspace(0, 2 * np.pi, n_timesteps)) + 0.05 * (aug_v % 5)
    timestamps = t_idx * dt
    steps = [
        Timestep(
            position=positions[i],
            orientation=orientations[i],
            force=float(forces[i]),
            timestamp=float(timestamps[i]),
        )
        for i in range(n_timesteps)
    ]
    return Trajectory(
        timesteps=steps,
        subject_id=subject_id,
        episode_id=episode_id,
        meta=dict(meta or {}),
    )


def trajectory_from_class_label(
    label: int,
    movements: list[dict],
    *,
    n_timesteps: int = 48,
    dt: float = 0.02,
    aug_v: int = 0,
    sample_id: int = 0,
    subject_id: str = "",
    episode_id: str = "",
    extra_meta: dict[str, Any] | None = None,
) -> Trajectory:
    da, dl, mag = movement_triple_for_label(movements, label)
    meta = {"label": label, **(extra_meta or {})}
    return trajectory_from_delta_triple(
        da,
        dl,
        mag,
        n_timesteps=n_timesteps,
        dt=dt,
        aug_v=aug_v,
        sample_id=sample_id,
        subject_id=subject_id,
        episode_id=episode_id,
        meta=meta,
    )
