"""
End-to-end: synthetic or file-backed data, subject split, run suite, optional I/O.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robot_policy_eval.config import ExperimentConfig
from robot_policy_eval.data.dataset import TrajectoryDataset
from robot_policy_eval.data.synthetic import generate_synthetic_dataset
from robot_policy_eval.experiments.runner import comparison_table_rows, run_evaluation_suite
from robot_policy_eval.paper.generate_capability_table import generate_capability_from_suite
from robot_policy_eval.paper.generate_tables import DEFAULT_POLICY_COLUMNS, generate_from_suite
from robot_policy_eval.policies.dummy_physio import DummyPhysioPolicy
from robot_policy_eval.policies.dummy_vla import DummyVLAPolicy
from robot_policy_eval.policies.hybrid import HybridPolicy


def default_policies(rng: np.random.Generator) -> list:
    physio = DummyPhysioPolicy(rng=np.random.default_rng(int(rng.integers(1 << 30))))
    vla = DummyVLAPolicy(stride=2, rng=np.random.default_rng(int(rng.integers(1 << 30))))
    hybrid = HybridPolicy(
        high_level=DummyVLAPolicy(stride=2, rng=np.random.default_rng(int(rng.integers(1 << 30)))),
        refiner=DummyPhysioPolicy(rng=np.random.default_rng(int(rng.integers(1 << 30)))),
    )
    return [physio, vla, hybrid]


def run_default_experiment(
    cfg: ExperimentConfig | None = None,
    n_subjects: int = 10,
    test_subject_ids: list[str] | None = None,
) -> tuple[dict, TrajectoryDataset, TrajectoryDataset]:
    """
    Builds synthetic data, holds out test subjects (default last 3 subjects),
    runs evaluation suite on **test** trajectories only (generalization).
    """
    cfg = cfg or ExperimentConfig()
    rng = np.random.default_rng(cfg.random_seed)
    full = generate_synthetic_dataset(n_subjects=n_subjects, rng=rng)
    subjects = full.unique_subjects()
    if test_subject_ids is None:
        test_subject_ids = subjects[-max(1, len(subjects) // 3) :]
    train_subjects = [s for s in subjects if s not in set(test_subject_ids)]
    train_ds, test_ds = full.split_by_subjects(train_subjects, test_subject_ids)

    policies = default_policies(rng)
    suite = run_evaluation_suite(policies, test_ds, cfg, rng=rng)

    meta = {
        "train_subjects": train_subjects,
        "test_subjects": test_subject_ids,
        "n_train_episodes": len(train_ds),
        "n_test_episodes": len(test_ds),
    }
    suite["split"] = meta
    suite["policy_columns"] = list(DEFAULT_POLICY_COLUMNS)
    return suite, train_ds, test_ds


def save_outputs(
    suite: dict,
    out_dir: Path,
    cfg: ExperimentConfig,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics_full.json", "w") as f:
        json.dump(suite, f, indent=2)

    # CSV for normal condition
    import csv

    rows = comparison_table_rows(suite, condition="normal")
    if rows:
        keys = list(rows[0].keys())
        with open(out_dir / "table_normal.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    # Wide CSV: all conditions per policy
    ag = suite.get("aggregates", {})
    flat = []
    for pol, conds in ag.items():
        for cname, metrics in conds.items():
            flat.append({"policy": pol, "condition": cname, **metrics})
    if flat:
        keys = list(flat[0].keys())
        with open(out_dir / "table_all_conditions.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(flat)

    policy_columns = suite.get("policy_columns")
    if policy_columns:
        columns = list(policy_columns)
    else:
        columns = None

    # Paper comparative table (Markdown + LaTeX + JSON) from same suite
    try:
        generate_from_suite(suite, out_dir, columns=columns)
    except (KeyError, ValueError) as e:
        import warnings

        warnings.warn(f"Paper table generation skipped: {e}", stacklevel=2)

    # Axis-grouped capability table (same metrics, framing for VLAs vs physio)
    try:
        generate_capability_from_suite(suite, out_dir, columns=columns)
    except (KeyError, ValueError) as e:
        import warnings

        warnings.warn(f"Capability table generation skipped: {e}", stacklevel=2)
