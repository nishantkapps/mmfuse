"""
Cross-model evaluation on **your** trajectory dataset (JSON).

Every policy is judged with the same `run_evaluation_suite` metrics (success, L2, jerk,
force, robustness, subject-split generalization). See `paper/CROSS_MODEL_EVAL.md`.
"""

from robot_policy_eval.cross_model.run import (
    evaluate_policies_on_trajectory_dataset,
    load_trajectory_dataset_and_split,
)

__all__ = [
    "evaluate_policies_on_trajectory_dataset",
    "load_trajectory_dataset_and_split",
]
