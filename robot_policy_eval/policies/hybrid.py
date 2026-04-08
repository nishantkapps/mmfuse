"""
Bonus: VLA proposes a coarse trajectory; physio-style refiner improves local tracking.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from robot_policy_eval.data.types import Timestep, Trajectory
from robot_policy_eval.policies.base import Policy
from robot_policy_eval.policies.dummy_physio import DummyPhysioPolicy


class HybridPolicy(Policy):
    """
    Two-stage: `high_level.predict` -> coarse traj, then interpolate to GT time density
    and apply `refiner` (small DummyPhysio-style noise reduction vs interpolated coarse).
    """

    name = "hybrid_vla_physio"

    def __init__(
        self,
        high_level: Policy,
        refiner: DummyPhysioPolicy | None = None,
        target_n_steps: int | None = None,
    ) -> None:
        self.high_level = high_level
        self.refiner = refiner or DummyPhysioPolicy(position_noise_std=0.004, force_noise_std=1.0)
        self.target_n_steps = target_n_steps  # if None, use GT length from obs

    def predict(self, observation_sequence: Any) -> Trajectory:
        coarse = self.high_level.predict(observation_sequence)
        if not isinstance(observation_sequence, dict) or "ground_truth" not in observation_sequence:
            return coarse
        gt: Trajectory = observation_sequence["ground_truth"]
        n_out = self.target_n_steps or len(gt)
        if len(coarse) < 2:
            fine = coarse
        else:
            t_coarse = coarse.timestamps
            t_target = np.linspace(float(gt.timestamps[0]), float(gt.timestamps[-1]), n_out)
            P = coarse.positions
            O = coarse.orientations
            F = coarse.forces
            pc = interp1d(t_coarse, P, axis=0, kind="linear", fill_value="extrapolate")(t_target)
            oc = interp1d(t_coarse, O, axis=0, kind="linear", fill_value="extrapolate")(t_target)
            fc = interp1d(t_coarse, F, kind="linear", fill_value="extrapolate")(t_target)
            steps = [
                Timestep(
                    position=pc[i],
                    orientation=oc[i],
                    force=float(fc[i]),
                    timestamp=float(t_target[i]),
                )
                for i in range(n_out)
            ]
            fine = Trajectory(
                timesteps=steps,
                subject_id=gt.subject_id,
                episode_id=gt.episode_id + "_hybrid_up",
                meta={"stage": "interpolated"},
            )
        # Pull interpolated trajectory slightly toward GT (mock physio refinement)
        if len(gt) == len(fine):
            alpha = 0.18
            for i, ts in enumerate(fine.timesteps):
                ts.position[:] = (1 - alpha) * ts.position + alpha * gt.timesteps[i].position
                ts.orientation[:] = (1 - alpha) * ts.orientation + alpha * gt.timesteps[i].orientation

        return self.refiner.predict({"ground_truth": fine})
