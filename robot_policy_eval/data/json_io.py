"""Load/save `TrajectoryDataset` from JSON (list of trajectory dicts)."""
from __future__ import annotations

import json
from pathlib import Path

from robot_policy_eval.data.dataset import TrajectoryDataset
from robot_policy_eval.data.types import Trajectory


def load_dataset_json(path: Path | str) -> TrajectoryDataset:
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "trajectories" in raw:
        raw = raw["trajectories"]
    trajs = [Trajectory.from_dict(x) for x in raw]
    return TrajectoryDataset(trajs, name=path.stem)


def save_dataset_json(dataset: TrajectoryDataset, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"name": dataset.name, "trajectories": [t.to_dict() for t in dataset.trajectories]}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
