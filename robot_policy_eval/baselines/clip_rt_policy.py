"""CLIP-RT (OpenCLIP ViT-H-14) → argmax over 8 commands → trajectory."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from robot_policy_eval.policies.base import Policy
from robot_policy_eval.trajectory_synthesis import trajectory_from_class_label

CLIP_RT_ARCH = "ViT-H-14-378-quickgelu"
DEFAULT_HF_REPO = "clip-rt/clip-rt-oxe-pretrained"
DEFAULT_HF_FILE = "cliprt-oxe-pretrained.pt"


def _pil_from_video_mid(video_path: str | Path):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return __import__("PIL.Image", fromlist=["Image"]).Image.new("RGB", (224, 224), (0, 0, 0))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


class ClipRTTrajectoryPolicy(Policy):
    name = "clip_rt"

    def __init__(
        self,
        movements: list[dict],
        *,
        repo_root: Path | str,
        device: str | None = None,
        hf_repo: str | None = None,
        hf_file: str | None = None,
        instruction: str | None = None,
        dual_cam: bool | None = None,
    ) -> None:
        self.movements = movements
        self.repo_root = Path(repo_root).resolve()
        self.device = device
        self.hf_repo = hf_repo or os.environ.get("CLIP_RT_HF_REPO", DEFAULT_HF_REPO)
        self.hf_file = hf_file or os.environ.get("CLIP_RT_HF_FILE", DEFAULT_HF_FILE)
        self.instruction = instruction or os.environ.get(
            "CLIP_RT_INSTRUCTION", "the therapist's next massage command"
        )
        v = os.environ.get("SDATA_FROZEN_DUAL_CAM", "1").strip().lower()
        self.dual_cam = dual_cam if dual_cam is not None else v not in ("0", "false", "no", "off")
        self._model = None
        self._torch_device = None
        self._commands = [m.get("command", f"part{i}") for i, m in enumerate(movements)]

    def _ensure_hub_timeout(self) -> None:
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

    def _load_weights(self) -> str:
        lw = os.environ.get("CLIP_RT_LOCAL_WEIGHTS", "").strip()
        if lw:
            p = Path(lw).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(f"CLIP_RT_LOCAL_WEIGHTS not found: {p}")
            return str(p)
        self._ensure_hub_timeout()
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=self.hf_repo, filename=self.hf_file)

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import open_clip
        import torch

        self._torch_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        wpath = self._load_weights()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=CLIP_RT_ARCH,
            pretrained=wpath,
        )
        model = model.to(self._torch_device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(CLIP_RT_ARCH)
        prompt = "what motion should the robot arm perform to complete the instruction '{}'"
        inst_text = tokenizer([prompt.format(self.instruction)]).to(self._torch_device)
        actions_tok = tokenizer(self._commands).to(self._torch_device)
        with __import__("torch").no_grad():
            inst_ref = model.encode_text(inst_text)
            action_ref = model.encode_text(actions_tok)
            inst_ref /= inst_ref.norm(dim=-1, keepdim=True)
            action_ref /= action_ref.norm(dim=-1, keepdim=True)
        self._model = model
        self._preprocess = preprocess
        self._inst_ref = inst_ref
        self._action_ref = action_ref
        self._tokenizer = tokenizer

    def _probs_from_pil(self, pil):
        import torch

        img = self._preprocess(pil).unsqueeze(0).to(self._torch_device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self._torch_device == "cuda")):
            img_feat = self._model.encode_image(img)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            ctx = img_feat + self._inst_ref
            ctx /= ctx.norm(dim=-1, keepdim=True)
            logits = ctx @ self._action_ref.T
            probs = logits.sigmoid()
        return probs.squeeze(0).float().cpu().numpy()

    def predict(self, observation_sequence: Any) -> Any:
        self._lazy_load()
        gt = observation_sequence["ground_truth"]
        meta = gt.meta or {}
        cam1 = meta.get("cam1")
        cam2 = meta.get("cam2", cam1)
        if not cam1:
            raise ValueError("clip_rt needs meta cam1")

        def resolve(p: str) -> Path:
            pp = Path(p)
            return pp if pp.is_file() else self.repo_root / p

        p1 = resolve(str(cam1))
        if self.dual_cam and cam2:
            p2 = resolve(str(cam2))
            probs = (
                self._probs_from_pil(_pil_from_video_mid(p1)) + self._probs_from_pil(_pil_from_video_mid(p2))
            ) / 2.0
        else:
            probs = self._probs_from_pil(_pil_from_video_mid(p1))
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
            extra_meta={"policy": "clip_rt", "pred_idx": pred_idx},
        )
