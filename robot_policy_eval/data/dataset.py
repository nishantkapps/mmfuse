"""
Dataset container and subject-based train/test splits for generalization evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from robot_policy_eval.data.types import Trajectory


@dataclass
class TrajectoryDataset:
    """Collection of trajectories with subject IDs."""

    trajectories: list[Trajectory]
    name: str = "dataset"

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Trajectory:
        return self.trajectories[idx]

    def subject_ids(self) -> list[str]:
        return [t.subject_id for t in self.trajectories]

    def unique_subjects(self) -> list[str]:
        return sorted(set(self.subject_ids()))

    def split_by_subjects(
        self,
        train_subjects: Sequence[str],
        test_subjects: Sequence[str],
    ) -> tuple["TrajectoryDataset", "TrajectoryDataset"]:
        train_set = {str(s) for s in train_subjects}
        test_set = {str(s) for s in test_subjects}
        overlap = train_set & test_set
        if overlap:
            raise ValueError(f"Train and test subject sets overlap: {overlap}")

        tr_train = [t for t in self.trajectories if t.subject_id in train_set]
        tr_test = [t for t in self.trajectories if t.subject_id in test_set]
        return (
            TrajectoryDataset(tr_train, name=f"{self.name}_train"),
            TrajectoryDataset(tr_test, name=f"{self.name}_test"),
        )

    def filter(self, predicate: Callable[[Trajectory], bool]) -> TrajectoryDataset:
        return TrajectoryDataset([t for t in self.trajectories if predicate(t)], name=self.name + "_filtered")


def split_subjects_fraction(
    dataset: TrajectoryDataset,
    test_fraction: float,
    seed: int = 42,
) -> tuple[TrajectoryDataset, TrajectoryDataset]:
    """Hold out a fraction of *subjects* (not episodes) for test."""
    import random

    subjects = dataset.unique_subjects()
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_test = max(1, int(round(len(subjects) * test_fraction)))
    test_subjects = subjects[:n_test]
    train_subjects = subjects[n_test:]
    return dataset.split_by_subjects(train_subjects, test_subjects)
