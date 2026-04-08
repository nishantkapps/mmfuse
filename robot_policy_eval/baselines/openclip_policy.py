"""OpenCLIP zero-shot → argmax over command strings → trajectory."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from robot_policy_eval.policies.base import Policy
from robot_policy_eval.trajectory_synthesis import trajectory_from_class_label

PRESETS: dict[str, tuple[str, str]] = {
    "b32": ("ViT-B-32", "laion2b_e16"),
    "l14": ("ViT-L-14", "laion2b_s32b_b82k"),
    "b16": ("ViT-B-16", "laion2b_s34b_b88k"),
}


class OpenCLIPTrajectoryPolicy(Policy):
    def __init__(
        self,
        movements: list[dict],
        *,
        repo_root: Path | str,
        preset: str = "b32",
        device: str | None = None,
        dual_cam: bool | None = None,
        name: str | None = None,
    ) -> None:
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {list(PRESETS)}")
        self.movements = movements
        self.repo_root = Path(repo_root).resolve()
        self.preset = preset
        self.device = device
        self.name = name or f"openclip_{preset}"
        v = os.environ.get("SDATA_FROZEN_DUAL_CAM", "1").strip().lower()
        self.dual_cam = dual_cam if dual_cam is not None else v not in ("0", "false", "no", "off")
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_feat = None
        self._logit_scale = None
        self._commands = [m.get("command", f"part{i}") for i, m in enumerate(movements)]
        tpl = os.environ.get("OPENCLIP_ZS_TEXT_TEMPLATE", "{}")
        if "{}" not in tpl:
            tpl = "{}"
        self._text_template = tpl

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
        import open_clip
        import torch

        mn, pt = PRESETS[self.preset]
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_device = dev
        model, _, preprocess = open_clip.create_model_and_transforms(mn, pretrained=pt)
        model = model.to(dev)
        model.eval()
        tokenizer = open_clip.get_tokenizer(mn)
        texts = [self._text_template.format(c) for c in self._commands]
        text_tok = tokenizer(texts).to(dev)
        with torch.no_grad():
            tf = model.encode_text(text_tok)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            ls = model.logit_scale.exp()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._text_feat = tf
        self._logit_scale = ls

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

    def _probs(self, pil):
        import torch

        img = self._preprocess(pil).unsqueeze(0).to(self._torch_device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self._torch_device == "cuda")):
            imf = self._model.encode_image(img)
            imf = imf / imf.norm(dim=-1, keepdim=True)
            logits = self._logit_scale * (imf @ self._text_feat.T)
        return logits.squeeze(0).float().softmax(dim=-1).cpu().numpy()

    def predict(self, observation_sequence: Any) -> Any:
        self._lazy_load()
        gt = observation_sequence["ground_truth"]
        meta = gt.meta or {}
        cam1 = meta.get("cam1")
        cam2 = meta.get("cam2", cam1)
        if not cam1:
            raise ValueError("openclip needs meta cam1")

        def resolve(p: str) -> Path:
            pp = Path(p)
            return pp if pp.is_file() else self.repo_root / p

        p1 = resolve(str(cam1))
        if self.dual_cam and cam2:
            p2 = resolve(str(cam2))
            probs = (self._probs(self._pil_mid(p1)) + self._probs(self._pil_mid(p2))) / 2.0
        else:
            probs = self._probs(self._pil_mid(p1))
        pred_idx = int(np.argmax(probs))
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
            extra_meta={"policy": self.name, "pred_idx": pred_idx},
        )
