"""
Task-specific controller: predicts continuous deltas then integrates to a trajectory (mock).

For reproducible evaluation, uses `ground_truth` inside `observation_sequence` when present
to generate a perturbed trajectory (simulates bounded tracking error).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from robot_policy_eval.data.types import Timestep, Trajectory
from robot_policy_eval.policies.base import Policy


class DummyPhysioPolicy(Policy):
    """Low-level policy: small additive noise on GT path (mock bounded-error tracking)."""

    name = "dummy_physio"

    def __init__(
        self,
        position_noise_std: float = 0.008,
        orientation_noise_std: float = 0.004,
        force_noise_std: float = 1.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.position_noise_std = position_noise_std
        self.orientation_noise_std = orientation_noise_std
        self.force_noise_std = force_noise_std
        self.rng = rng or np.random.default_rng()

    def predict(self, observation_sequence: Any) -> Trajectory:
        if not isinstance(observation_sequence, dict) or "ground_truth" not in observation_sequence:
            raise ValueError(
                "DummyPhysioPolicy expects observation_sequence['ground_truth'] as Trajectory "
                "(swap for real proprio/images in production)."
            )
        gt: Trajectory = observation_sequence["ground_truth"]
        steps: list[Timestep] = []
        for ts in gt.timesteps:
            p = ts.position + self.rng.normal(0, self.position_noise_std, 3)
            o = ts.orientation + self.rng.normal(0, self.orientation_noise_std, 3)
            f = float(ts.force + self.rng.normal(0, self.force_noise_std))
            steps.append(Timestep(position=p, orientation=o, force=f, timestamp=ts.timestamp))
        return Trajectory(
            timesteps=steps,
            subject_id=gt.subject_id,
            episode_id=gt.episode_id + "_physio",
            meta={**gt.meta, "policy": self.name},
        )
