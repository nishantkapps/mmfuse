"""MMFuse checkpoint → movement triple → same trajectory synthesis as GT."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from robot_policy_eval.policies.base import Policy
from robot_policy_eval.trajectory_synthesis import trajectory_from_delta_triple


def _ensure_repo_on_path(repo_root: Path | None) -> Path:
    root = repo_root or Path(os.environ.get("MMFUSE_REPO_ROOT", Path.cwd()))
    root = Path(root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _load_predict_module(repo_root: Path):
    path = repo_root / "scripts" / "predict_with_model.py"
    if not path.is_file():
        raise FileNotFoundError(f"Expected {path} (set MMFUSE_REPO_ROOT to repo root).")
    spec = importlib.util.spec_from_file_location("mmfuse_predict_with_model", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class MMFuseTrajectoryPolicy(Policy):
    """
    Loads `scripts.predict_with_model` (same fusion + movement head as training).
    Requires: trajectory `meta` with `cam1`, `cam2`, `audio` (paths relative to repo or absolute).
    """

    name = "mmfuse"

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        repo_root: Path | str | None = None,
        mode: str = "full",
        device: str | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.repo_root = _ensure_repo_on_path(Path(repo_root) if repo_root else None)
        self.mode = mode
        self.device = device
        self._encoders = None
        self._head = None
        self._labels = None
        self._pm = None

    def _lazy_load(self) -> None:
        if self._encoders is not None:
            return
        pm = _load_predict_module(self.repo_root)
        self._pm = pm
        dev = self.device or pm.setup_environment()
        encoders = pm.load_encoders(dev)
        head, labels = pm.load_checkpoint(dev, encoders, str(self.checkpoint))
        self._encoders = encoders
        self._head = head
        self._labels = labels
        self._device = dev

    def predict(self, observation_sequence: Any) -> Any:
        self._lazy_load()
        assert self._pm is not None
        gt: Trajectory = observation_sequence["ground_truth"]
        meta = gt.meta or {}
        cam1 = meta.get("cam1")
        cam2 = meta.get("cam2", cam1)
        audio = meta.get("audio", "")
        if not cam1:
            raise ValueError("MMFuseTrajectoryPolicy needs ground_truth.meta['cam1'] (SData export).")

        def _resolve(p: str) -> str | None:
            if not p:
                return None
            pp = Path(p)
            if pp.is_file():
                return str(pp)
            cand = self.repo_root / p
            return str(cand) if cand.is_file() else None

        c1 = _resolve(str(cam1))
        c2 = _resolve(str(cam2)) if cam2 else None
        wav = _resolve(str(audio)) if audio else None
        if self.mode == "vision_only":
            wav = None

        fused = self._pm._build_fused(self._encoders, c1, c2, wav, self._device)
        da, dl, mag = self._pm._predict_movement(fused, self._encoders)
        aug_v = int(meta.get("aug_v", 0))
        sid = int(meta.get("sample_id", 0))
        n = max(2, len(gt))
        ts = gt.timestamps
        dt = float(np.mean(np.diff(ts))) if len(ts) > 1 else 0.02
        return trajectory_from_delta_triple(
            da,
            dl,
            mag,
            n_timesteps=n,
            dt=dt,
            aug_v=aug_v,
            sample_id=sid,
            subject_id=gt.subject_id,
            episode_id=gt.episode_id,
            meta={"policy": "mmfuse", "pred_delta": (da, dl, mag)},
        )
