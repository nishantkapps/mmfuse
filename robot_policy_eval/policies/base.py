"""
Abstract policy interface: map an observation sequence to a predicted trajectory.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from robot_policy_eval.data.types import Trajectory


class Policy(ABC):
    """
    Heterogeneous policies (delta controllers, VLAs, diffusion) expose the same call.

    `observation_sequence` is intentionally untyped (dict, list, ndarray) so real
    implementations can pass images, proprio, language, etc.
    """

    name: str = "policy"

    @abstractmethod
    def predict(self, observation_sequence: Any) -> Trajectory:
        """Return a predicted trajectory (same Timestep schema as ground truth)."""
        raise NotImplementedError
