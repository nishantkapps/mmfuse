"""
Default thresholds and evaluation hyperparameters (override via ExperimentConfig).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    """
    No magic numbers in runner code — tune here or pass overrides.

    **Future perturbations (same metric stack):** To add domain-specific robustness tests
    (e.g., lighting jitter, synthetic human motion), extend the observation builder in
    `experiments/runner.py` and list new condition names here — keep ground-truth
    trajectories clean for metric computation. See `paper/COMPARISON_FRAMEWORK.md`.
    """

    # Task success: mean per-timestep position error (m) below this counts toward success
    trajectory_success_rmse_threshold: float = 0.05
    # Safety
    force_safe_max: float = 50.0  # Newtons; exceed -> violation

    # Robustness sweeps
    position_noise_sigmas: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)
    occlusion_drop_probs: tuple[float, ...] = (0.0, 0.1, 0.2)

    # Resampling for different-length trajectories (metrics)
    metric_resample_points: int = 64

    # Output
    output_dir: str = "robot_policy_eval/outputs"
    random_seed: int = 42

    extra: dict[str, Any] = field(default_factory=dict)
