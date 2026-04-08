"""
Shared utilities: manifest I/O, frames, command strings from YAML, post-hoc text→label for reporting.

Post-hoc matching does NOT replace model forward; it only maps free-form model output to a class index
so we can compute accuracy vs manifest labels.
"""
from __future__ import annotations

import csv
import difflib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_manifest_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            row["label"] = int(row["label"])
            row["aug_v"] = int(row["aug_v"])
            row["sample_id"] = int(row["sample_id"])
            rows.append(row)
    return rows


def filter_split(rows: list[dict], split: str) -> list[dict]:
    if split == "all":
        return rows
    return [x for x in rows if x["split"] == split]


def load_sdata_command_strings(
    movement_yaml: Path | None = None,
) -> list[str]:
    """Eight command strings in part0..part7 order (same as training YAML)."""
    if movement_yaml is None:
        movement_yaml = REPO_ROOT / "config" / "sdata_movement_config.yaml"
    with open(movement_yaml) as f:
        cfg = yaml.safe_load(f)
    movements = cfg.get("movements", [])
    return [m.get("command", f"part{i}") for i, m in enumerate(movements)]


def pil_from_video_mid_frame(video_path: str | Path):
    """Middle frame as PIL RGB (native input for most VLAs)."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return Image.new("RGB", (224, 224), color=(0, 0, 0))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def env_frozen_dual_cam() -> bool:
    """
    When true (default), frozen CLIP-style baselines average scores over cam1 + cam2 mid-frames,
    aligning with MMFuse which uses both camera streams (still one mid-frame per video).
    Set SDATA_FROZEN_DUAL_CAM=0 to use a single camera only (legacy / ablation).
    """
    import os

    v = os.environ.get("SDATA_FROZEN_DUAL_CAM", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def posthoc_text_to_label(raw_text: str, commands: list[str]) -> tuple[int, str]:
    """
    Map model-generated text to the index of the best-matching SData command (reporting only).
    Returns (label_idx, method) where method describes the match heuristic.
    """
    raw = (raw_text or "").strip()
    if not raw:
        return 0, "empty_fallback"
    raw_l = raw.lower()
    for i, c in enumerate(commands):
        cl = c.lower()
        if cl in raw_l or raw_l in cl:
            return i, "substring"
    # Fuzzy pick among full command strings
    best = difflib.get_close_matches(raw, commands, n=1, cutoff=0.25)
    if best:
        return commands.index(best[0]), "fuzzy_full"
    # Line-wise: take first line and fuzzy match
    first = raw.split("\n")[0].strip()
    best2 = difflib.get_close_matches(first, commands, n=1, cutoff=0.2)
    if best2:
        return commands.index(best2[0]), "fuzzy_first_line"
    # Token overlap score
    scores = []
    for i, c in enumerate(commands):
        ws = set(raw_l.split())
        cs = set(c.lower().split())
        inter = len(ws & cs)
        scores.append((inter, i))
    scores.sort(reverse=True)
    return scores[0][1], "token_overlap"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def metrics_from_preds(y_true: list[int], y_pred: list[int], num_classes: int = 8) -> dict[str, Any]:
    try:
        from sdata_vla_benchmark.metrics.classification import summarize
    except ImportError:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "sdata_metrics", REPO_ROOT / "sdata_vla_benchmark" / "metrics" / "classification.py"
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        summarize = mod.summarize

    return summarize(y_true, y_pred, num_classes=num_classes)
