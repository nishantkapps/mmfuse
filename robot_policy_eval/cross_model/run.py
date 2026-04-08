"""
Evaluate one or more `Policy` implementations on the same held-out trajectories.

This is the intended way to compare **your** controller against VLAs or hybrids on **your**
dataset: every model must implement `predict(observation) -> Trajectory`; metrics are
identical (see `paper/COMPARISON_FRAMEWORK.md`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_policy_eval.config import ExperimentConfig
from robot_policy_eval.data.dataset import TrajectoryDataset, split_subjects_fraction
from robot_policy_eval.data.json_io import load_dataset_json
from robot_policy_eval.experiments.pipeline import save_outputs
from robot_policy_eval.experiments.runner import run_evaluation_suite
from robot_policy_eval.policies.base import Policy


def load_trajectory_dataset_and_split(
    dataset_path: Path | str,
    *,
    test_subject_ids: list[str] | None = None,
    test_fraction: float | None = None,
    seed: int = 42,
) -> tuple[TrajectoryDataset, TrajectoryDataset]:
    """
    Load `TrajectoryDataset` from JSON and split by **subject** for generalization.

    - If `test_subject_ids` is set, those subjects form the test set; remaining subjects train.
    - Else if `test_fraction` is set, that fraction of subjects (shuffled) is test.
    - Else: hold out the last ~1/3 of subjects (sorted by `unique_subjects()`).
    """
    full = load_dataset_json(dataset_path)
    subjects = full.unique_subjects()
    if not subjects:
        raise ValueError(
            "Dataset has no subject_id on trajectories; set subject_id on each Trajectory for generalization splits."
        )

    if test_subject_ids is not None:
        ts = [str(s) for s in test_subject_ids]
        train_subjects = [s for s in subjects if s not in set(ts)]
        missing = set(ts) - set(subjects)
        if missing:
            raise ValueError(f"test_subject_ids not found in dataset: {sorted(missing)}")
        return full.split_by_subjects(train_subjects, ts)

    if test_fraction is not None:
        return split_subjects_fraction(full, test_fraction, seed=seed)

    test_n = max(1, len(subjects) // 3)
    test_subjects = subjects[-test_n:]
    train_subjects = subjects[:-test_n]
    return full.split_by_subjects(train_subjects, test_subjects)


def evaluate_policies_on_trajectory_dataset(
    policies: list[Policy],
    dataset_path: Path | str,
    cfg: ExperimentConfig | None = None,
    *,
    test_subject_ids: list[str] | None = None,
    test_fraction: float | None = None,
    seed: int = 42,
) -> tuple[dict, TrajectoryDataset, TrajectoryDataset]:
    """
    Run `run_evaluation_suite` on the **test** split only; return suite dict + train/test datasets.

    Call `save_outputs(suite, out_dir, cfg)` to write CSV/JSON/paper tables.
    """
    train_ds, test_ds = load_trajectory_dataset_and_split(
        dataset_path,
        test_subject_ids=test_subject_ids,
        test_fraction=test_fraction,
        seed=seed,
    )
    cfg = cfg or ExperimentConfig(random_seed=seed)
    rng = np.random.default_rng(seed)
    suite = run_evaluation_suite(policies, test_ds, cfg, rng=rng)
    suite["split"] = {
        "train_subjects": train_ds.unique_subjects(),
        "test_subjects": test_ds.unique_subjects(),
        "n_train_episodes": len(train_ds),
        "n_test_episodes": len(test_ds),
        "dataset_path": str(Path(dataset_path).resolve()),
    }
    from robot_policy_eval.baselines.factory import DISPLAY_NAMES_DEFAULT

    suite["policy_columns"] = [(p.name, DISPLAY_NAMES_DEFAULT.get(p.name, p.name)) for p in policies]
    return suite, train_ds, test_ds


def run_from_cli(
    dataset_path: Path,
    output_dir: Path,
    policy_names: list[str],
    *,
    test_subject_ids: list[str] | None = None,
    test_fraction: float | None = None,
    seed: int = 42,
    no_plots: bool = False,
) -> dict:
    """CLI helper: built-in policies + save_outputs. Returns suite dict."""
    from robot_policy_eval.cross_model.builtins import make_builtin_policies

    cfg = ExperimentConfig(random_seed=seed, output_dir=str(output_dir))
    policies = make_builtin_policies(policy_names, seed=seed)
    suite, _, test_ds = evaluate_policies_on_trajectory_dataset(
        policies,
        dataset_path,
        cfg,
        test_subject_ids=test_subject_ids,
        test_fraction=test_fraction,
        seed=seed,
    )
    # policy_columns already set in evaluate_policies_on_trajectory_dataset
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_outputs(suite, output_dir, cfg)

    if not no_plots and len(test_ds) > 0:
        _write_optional_plots(suite, policies, test_ds, output_dir, seed)

    return suite


def _write_optional_plots(
    suite: dict,
    policies: list,
    test_ds: TrajectoryDataset,
    output_dir: Path,
    seed: int,
) -> None:
    import numpy as np

    from robot_policy_eval.visualization.plots import (
        plot_3d_three_trajectories,
        plot_3d_trajectories,
        plot_error_vs_time,
        plot_error_vs_timestep_three_models,
        plot_occlusion_curve,
        plot_robustness_curve,
    )

    rng = np.random.default_rng(seed)
    gt0 = test_ds[0]
    obs = {"ground_truth": gt0}
    preds_by_name = {pol.name: pol.predict(obs) for pol in policies}

    physio = preds_by_name.get("dummy_physio")
    vla = preds_by_name.get("dummy_vla")
    hybrid = preds_by_name.get("hybrid_vla_physio") or preds_by_name.get("hybrid_demo")
    if physio is not None and vla is not None:
        plot_3d_three_trajectories(
            gt0,
            physio,
            vla,
            output_dir / "fig_paper_trajectory_gt_physio_vla.png",
            hybrid=hybrid,
            title="Trajectory comparison: GT vs Physio vs VLA",
        )
    elif len(policies) >= 2:
        a, b = policies[0], policies[1]
        h = preds_by_name.get(policies[2].name) if len(policies) > 2 else None
        plot_3d_three_trajectories(
            gt0,
            preds_by_name[a.name],
            preds_by_name[b.name],
            output_dir / "fig_paper_trajectory_gt_physio_vla.png",
            hybrid=h,
            title="Trajectory comparison: GT vs models",
            label_a=a.name,
            label_b=b.name,
            label_hybrid=policies[2].name if len(policies) > 2 else "Hybrid",
        )
    label_map = {
        "dummy_physio": "Physio (ours)",
        "dummy_vla": "VLA",
        "hybrid_vla_physio": "Hybrid",
    }
    preds_labeled = {label_map.get(k, k): v for k, v in preds_by_name.items()}
    plot_error_vs_timestep_three_models(
        gt0,
        preds_labeled,
        output_dir / "fig_paper_error_vs_timestep.png",
    )
    for pol in policies:
        pred = preds_by_name[pol.name]
        plot_3d_trajectories(
            gt0,
            pred,
            output_dir / f"fig_3d_{pol.name}.png",
            title=f"{pol.name}: GT vs pred",
        )
        plot_error_vs_time(gt0, pred, output_dir / f"fig_err_time_{pol.name}.png")
    plot_robustness_curve(suite, output_dir / "fig_robustness_noise.png")
    plot_occlusion_curve(suite, output_dir / "fig_robustness_occlusion.png")
