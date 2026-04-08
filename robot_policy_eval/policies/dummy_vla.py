"""
VLA-style policy: returns a coarser / temporally subsampled trajectory (mock high-level plan).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from robot_policy_eval.data.types import Timestep, Trajectory
from robot_policy_eval.policies.base import Policy


class DummyVLAPolicy(Policy):
    """
    Mock VLA: subsample GT, add bias/smoothing — simulates a coarse plan that misses fine detail.
    """

    name = "dummy_vla"

    def __init__(
        self,
        stride: int = 2,
        spatial_bias: np.ndarray | None = None,
        smooth_sigma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.stride = max(1, stride)
        self.spatial_bias = spatial_bias if spatial_bias is not None else np.array([0.02, -0.01, 0.015])
        self.smooth_sigma = smooth_sigma
        self.rng = rng or np.random.default_rng()

    def predict(self, observation_sequence: Any) -> Trajectory:
        if not isinstance(observation_sequence, dict) or "ground_truth" not in observation_sequence:
            raise ValueError("DummyVLAPolicy expects observation_sequence['ground_truth'] as Trajectory")
        gt: Trajectory = observation_sequence["ground_truth"]
        idx = np.arange(0, len(gt), self.stride, dtype=int)
        if len(idx) == 0:
            idx = np.array([0])
        steps: list[Timestep] = []
        for i in idx:
            ts = gt.timesteps[i]
            p = ts.position.copy() + self.spatial_bias + self.rng.normal(0, 0.012, 3)
            o = ts.orientation.copy() + self.rng.normal(0, 0.02, 3)
            f = float(ts.force + self.rng.normal(0, 3.0))
            steps.append(Timestep(position=p, orientation=o, force=f, timestamp=ts.timestamp))

        # Optional smoothing along the downsampled sequence
        if len(steps) >= 3 and self.smooth_sigma > 0:
            P = np.stack([s.position for s in steps], axis=0)
            for d in range(3):
                P[:, d] = gaussian_filter1d(P[:, d], sigma=self.smooth_sigma, mode="nearest")
            for i, s in enumerate(steps):
                s.position = P[i]

        return Trajectory(
            timesteps=steps,
            subject_id=gt.subject_id,
            episode_id=gt.episode_id + "_vla",
            meta={**gt.meta, "policy": self.name, "stride": self.stride},
        )
