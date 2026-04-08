"""Placeholders for stacks not wired to continuous EE control (metrics still run)."""
from __future__ import annotations

from typing import Any

import numpy as np

from robot_policy_eval.policies.base import Policy
from robot_policy_eval.trajectory_synthesis import trajectory_from_delta_triple


class UnintegratedStackPolicy(Policy):
    """
    Returns a valid trajectory from fixed deltas so L2/jerk/success are defined.
    `meta.stub_reason` documents that this is not the real system output.
    """

    def __init__(self, internal_name: str, *, delta: tuple[float, float, float] = (0.5, -0.5, 1.0)) -> None:
        self.name = internal_name
        self._delta = delta

    def predict(self, observation_sequence: Any) -> Any:
        gt = observation_sequence["ground_truth"]
        n = max(2, len(gt))
        ts = gt.timestamps
        dt = float(np.mean(np.diff(ts))) if len(ts) > 1 else 0.02
        aug_v = int((gt.meta or {}).get("aug_v", 0))
        sid = int((gt.meta or {}).get("sample_id", 0))
        return trajectory_from_delta_triple(
            self._delta[0],
            self._delta[1],
            self._delta[2],
            n_timesteps=n,
            dt=dt,
            aug_v=aug_v,
            sample_id=sid,
            subject_id=gt.subject_id,
            episode_id=gt.episode_id,
            meta={
                "stub": True,
                "stub_reason": "Real-time inference for this stack is not integrated; placeholder deltas only.",
            },
        )
