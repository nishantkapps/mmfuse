"""Classification metrics for benchmark JSON outputs."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    a = np.asarray(list(y_true))
    b = np.asarray(list(y_pred))
    if len(a) == 0:
        return 0.0
    return float((a == b).mean())


def macro_f1(y_true: List[int], y_pred: List[int], num_classes: int | None = None) -> float:
    try:
        from sklearn.metrics import f1_score
    except ImportError as e:
        raise ImportError("scikit-learn required: pip install scikit-learn") from e
    if num_classes is None:
        num_classes = max(max(y_true or [0]), max(y_pred or [0])) + 1
    return float(
        f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)), zero_division=0)
    )


def summarize(y_true: List[int], y_pred: List[int], num_classes: int = 8) -> dict:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes=num_classes),
        "n": len(y_true),
    }
