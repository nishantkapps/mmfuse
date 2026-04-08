"""
Contract for wrapping external VLAs so they participate in the same evaluation as physio policies.

All comparisons in `run_evaluation_suite` go through `Policy.predict(...) -> Trajectory`.
VLAs rarely emit that structure natively; this module documents the mapping and provides
helpers — **no** imports from other project folders.

See `paper/COMPARISON_FRAMEWORK.md` for why discrete-only scores are not headline metrics.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from robot_policy_eval.data.types import Timestep, Trajectory
from robot_policy_eval.policies.base import Policy


def stack_to_trajectory(
    positions: np.ndarray,
    *,
    orientations: np.ndarray | None = None,
    forces: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    subject_id: str = "",
    episode_id: str = "",
    meta: dict[str, Any] | None = None,
) -> Trajectory:
    """
    Build a `Trajectory` from aligned arrays (e.g. after resampling a VLA waypoint sequence).

    - `positions`: (T, 3). Required.
    - `orientations`: (T, 3) RPY rad; default zeros if None.
    - `forces`: (T,) or scalar broadcast; default zeros if None.
    - `timestamps`: length T; default uniform 0..T-1 if None.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (T, 3), got {pos.shape}")
    t = pos.shape[0]
    if t == 0:
        return Trajectory(timesteps=[], subject_id=subject_id, episode_id=episode_id, meta=dict(meta or {}))

    if orientations is None:
        ori = np.zeros((t, 3), dtype=np.float64)
    else:
        ori = np.asarray(orientations, dtype=np.float64).reshape(t, 3)

    if forces is None:
        f = np.zeros(t, dtype=np.float64)
    else:
        f = np.asarray(forces, dtype=np.float64).reshape(-1)
        if f.size == 1:
            f = np.full(t, float(f[0]), dtype=np.float64)
        elif f.shape[0] != t:
            raise ValueError(f"forces length {f.shape[0]} != T={t}")

    if timestamps is None:
        ts = np.arange(t, dtype=np.float64)
    else:
        ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if ts.shape[0] != t:
            raise ValueError(f"timestamps length {ts.shape[0]} != T={t}")

    steps = [
        Timestep(position=pos[i], orientation=ori[i], force=float(f[i]), timestamp=float(ts[i]))
        for i in range(t)
    ]
    return Trajectory(
        timesteps=steps,
        subject_id=subject_id,
        episode_id=episode_id,
        meta=dict(meta or {}),
    )


class ExternalTrajectoryPolicy(Policy):
    """
    Base class for any wrapped model that ultimately produces a `Trajectory`.

    Subclasses implement `predict_trajectory`: map `observation_sequence` (images, proprio,
    language ids, etc.) into the shared representation. `predict` delegates to it so the
    runner does not need to know about VLA-specific types.

    **Proxy trajectories:** If you only have discrete commands or language, mapping them to
    a trajectory **without** a principled dynamical model is a *proxy* — useful for
    debugging or appendix only; state limitations in the paper (see COMPARISON_FRAMEWORK.md).
    """

    @abstractmethod
    def predict_trajectory(self, observation_sequence: Any) -> Trajectory:
        """Return prediction in the same schema as ground-truth trajectories."""

    def predict(self, observation_sequence: Any) -> Trajectory:
        return self.predict_trajectory(observation_sequence)


class ResampledVLAWrapper(ExternalTrajectoryPolicy):
    """
    Example stub: wrap an inner policy that already returns a `Trajectory` but with a
    different temporal length — identity pass-through.

    Replace this with your own subclass that calls a VLA, then `stack_to_trajectory`.
    """

    def __init__(self, inner: Policy, *, name: str = "vla_resampled") -> None:
        self._inner = inner
        self.name = name

    def predict_trajectory(self, observation_sequence: Any) -> Trajectory:
        return self._inner.predict(observation_sequence)
