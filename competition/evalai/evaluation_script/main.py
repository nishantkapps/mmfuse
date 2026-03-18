"""
MIRA Challenge — EvalAI Evaluation Script

EvalAI calls evaluate() with ground-truth and submission file paths.
This wrapper delegates to the same metric functions used in competition/evaluate.py.
"""

import csv
import json
import math
from collections import defaultdict

from sklearn.metrics import accuracy_score, f1_score

ACTION_LABELS = [
    "Start", "Go Here", "Move Down", "Move Up",
    "Stop", "Move Left", "Move Right", "Perfect",
]

MAX_MAE_CM = 20.0


def _load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "sample": r["sample"].strip(),
                "action": r["action"].strip(),
                "dx": float(r["dx"]),
                "dy": float(r["dy"]),
            })
    return {r["sample"]: r for r in rows}


def _track1_metrics(gt, pred):
    samples = sorted(gt.keys() & pred.keys())
    if not samples:
        return {"Macro_F1": 0.0, "Accuracy": 0.0}
    y_true = [gt[s]["action"] for s in samples]
    y_pred = [pred[s]["action"] for s in samples]
    return {
        "Macro_F1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 5),
        "Accuracy": round(accuracy_score(y_true, y_pred), 5),
    }


def _track2_metrics(gt, pred):
    samples = sorted(gt.keys() & pred.keys())
    if not samples:
        return {"MAE": 999.0}
    errors = []
    for s in samples:
        errors.append(abs(gt[s]["dx"] - pred[s]["dx"]) + abs(gt[s]["dy"] - pred[s]["dy"]))
    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
    return {
        "MAE": round(mae, 5),
        "RMSE": round(rmse, 5),
    }


def _combined_score(macro_f1, mae):
    norm_mae = min(mae / MAX_MAE_CM, 1.0) if MAX_MAE_CM > 0 else 0.0
    return round(0.60 * macro_f1 + 0.40 * (1.0 - norm_mae), 5)


def evaluate(test_annotation_file, user_annotation_file, phase_codename, **kwargs):
    """
    Entry point called by EvalAI.

    Phase codenames:
      - track1_dev / track1_test  -> Action Recognition
      - track2_dev / track2_test  -> Motion Prediction
      - combined_dev / combined_test -> Both tracks
    """
    gt = _load_csv(test_annotation_file)
    pred = _load_csv(user_annotation_file)

    t1 = _track1_metrics(gt, pred)
    t2 = _track2_metrics(gt, pred)
    score = _combined_score(t1["Macro_F1"], t2["MAE"])

    split_name = "test_split" if "test" in phase_codename else "dev_split"

    if phase_codename.startswith("track1"):
        return {"result": [{split_name: t1}]}

    if phase_codename.startswith("track2"):
        return {"result": [{split_name: t2}]}

    # Combined phase
    combined = {
        "Combined_Score": score,
        "Macro_F1": t1["Macro_F1"],
        "Accuracy": t1["Accuracy"],
        "MAE": t2["MAE"],
    }
    return {"result": [{split_name: combined}]}
