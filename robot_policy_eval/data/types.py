"""
Trajectory data format: each timestep has position, orientation (RPY), force, timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


@dataclass
class Timestep:
    """Single instant along an end-effector trajectory."""

    position: np.ndarray  # (3,) x,y,z
    orientation: np.ndarray  # (3,) roll, pitch, yaw (rad)
    force: float
    timestamp: float

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.orientation = np.asarray(self.orientation, dtype=np.float64).reshape(3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.tolist(),
            "orientation": self.orientation.tolist(),
            "force": float(self.force),
            "timestamp": float(self.timestamp),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Timestep:
        return cls(
            position=np.asarray(d["position"], dtype=np.float64),
            orientation=np.asarray(d["orientation"], dtype=np.float64),
            force=float(d["force"]),
            timestamp=float(d["timestamp"]),
        )


@dataclass
class Trajectory:
    """Ordered sequence of timesteps (variable length)."""

    timesteps: list[Timestep]
    subject_id: str = ""
    episode_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.timesteps)

    def __iter__(self) -> Iterator[Timestep]:
        return iter(self.timesteps)

    @property
    def positions(self) -> np.ndarray:
        if not self.timesteps:
            return np.zeros((0, 3))
        return np.stack([t.position for t in self.timesteps], axis=0)

    @property
    def orientations(self) -> np.ndarray:
        if not self.timesteps:
            return np.zeros((0, 3))
        return np.stack([t.orientation for t in self.timesteps], axis=0)

    @property
    def forces(self) -> np.ndarray:
        return np.array([t.force for t in self.timesteps], dtype=np.float64)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([t.timestamp for t in self.timesteps], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "episode_id": self.episode_id,
            "meta": dict(self.meta),
            "timesteps": [t.to_dict() for t in self.timesteps],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trajectory:
        return cls(
            timesteps=[Timestep.from_dict(x) for x in d["timesteps"]],
            subject_id=str(d.get("subject_id", "")),
            episode_id=str(d.get("episode_id", "")),
            meta=dict(d.get("meta", {})),
        )
