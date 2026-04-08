"""Hugging Face Vision2Seq (e.g. OpenVLA) → text → command index → trajectory."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from robot_policy_eval.baselines.text_to_label import posthoc_text_to_label
from robot_policy_eval.policies.base import Policy
from robot_policy_eval.trajectory_synthesis import trajectory_from_class_label


def _build_prompt(commands: list[str]) -> str:
    lines = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(commands))
    return (
        "You view a single camera image from a therapeutic massage session. "
        "Exactly one of the following commands applies as the intended next instruction.\n"
        f"{lines}\n"
        "Reply with only the exact command phrase (verbatim from the list) that best matches the scene."
    )


def _default_report_name(model_id: str) -> str:
    slug = model_id.rstrip("/").split("/")[-1]
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", slug)[:80]
    return slug or "hf_vla"


def _patch_attn_config(config) -> None:
    impl = os.environ.get("HF_VLA_ATTN_IMPLEMENTATION", "eager")
    stack = [config]
    seen: set[int] = set()
    while stack:
        cfg = stack.pop()
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        if not hasattr(cfg, "_attn_implementation"):
            setattr(cfg, "_attn_implementation", impl)
        for key in ("text_config", "vision_config", "llm_config", "encoder_config", "decoder_config"):
            sub = getattr(cfg, key, None)
            if sub is not None:
                stack.append(sub)


class HFVLAtrajectoryPolicy(Policy):
    """Loads AutoModelForVision2Seq + processor; one image (cam1) per step."""

    def __init__(
        self,
        movements: list[dict],
        *,
        repo_root: Path | str,
        model_id: str | None = None,
        device: str | None = None,
        name: str | None = None,
    ) -> None:
        self.movements = movements
        self.repo_root = Path(repo_root).resolve()
        mid = model_id or os.environ.get(
            "OPEN_VLA_MODEL_ID", os.environ.get("HF_VLA_MODEL_ID", "openvla/openvla-7b")
        )
        self.model_id = mid
        ra = (os.environ.get("HF_VLA_REPORT_AS", "") or "").strip()
        if ra:
            slug = ra
        elif "openvla" in mid.lower():
            slug = "openvla"
        else:
            slug = _default_report_name(mid)
        self.name = name or slug
        self.device = device
        self._processor = None
        self._model = None
        self._commands = [m.get("command", f"part{i}") for i, m in enumerate(movements)]

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        model_config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        _patch_attn_config(model_config)
        model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            config=model_config,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        model.eval()
        model.to(dev)
        self._processor = processor
        self._model = model
        self._torch_device = dev

    def _pil_mid(self, video_path: Path):
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return Image.new("RGB", (224, 224), (0, 0, 0))
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def predict(self, observation_sequence: Any) -> Any:
        self._lazy_load()
        import torch

        gt = observation_sequence["ground_truth"]
        meta = gt.meta or {}
        cam1 = meta.get("cam1")
        if not cam1:
            raise ValueError("openvla needs meta cam1")
        pp = Path(cam1)
        path = pp if pp.is_file() else self.repo_root / cam1
        image = self._pil_mid(path)
        prompt = _build_prompt(self._commands)
        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._torch_device)
        else:
            inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=int(os.environ.get("HF_VLA_MAX_NEW_TOKENS", "32")))
        text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        pred_idx, _ = posthoc_text_to_label(text, self._commands)
        aug_v = int(meta.get("aug_v", 0))
        sid = int(meta.get("sample_id", 0))
        n = max(2, len(gt))
        ts = gt.timestamps
        dt = float(np.mean(np.diff(ts))) if len(ts) > 1 else 0.02
        return trajectory_from_class_label(
            pred_idx,
            self.movements,
            n_timesteps=n,
            dt=dt,
            aug_v=aug_v,
            sample_id=sid,
            subject_id=gt.subject_id,
            episode_id=gt.episode_id,
            extra_meta={"policy": self.name, "raw_text": text[:500], "pred_idx": pred_idx},
        )
