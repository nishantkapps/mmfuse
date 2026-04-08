#!/usr/bin/env python3
"""
Frozen CLIP-RT — native inference exactly as in the paper / GitHub README:

  contrastive scores between (image + instruction) and each candidate command embedding,
  then argmax over candidates (their test-time procedure for selecting a motion primitive).

Weights: downloaded from Hugging Face (default: clip-rt/clip-rt-oxe-pretrained, file cliprt-oxe-pretrained.pt).

Candidate commands are the eight SData phrases from config/sdata_movement_config.yaml (same strings as your labels).
This is not a separate classifier head — it is how CLIP-RT selects among text-defined motions.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdata_vla_benchmark.frozen.common import (
    env_frozen_dual_cam,
    filter_split,
    load_manifest_rows,
    load_sdata_command_strings,
    metrics_from_preds,
    pil_from_video_mid_frame,
    write_json,
)

# Official CLIP-RT README model id (OpenCLIP)
CLIP_RT_ARCH = "ViT-H-14-378-quickgelu"
DEFAULT_HF_REPO = "clip-rt/clip-rt-oxe-pretrained"
DEFAULT_HF_FILE = "cliprt-oxe-pretrained.pt"

# HF default read timeout is 10s — large .pt files often exceed that on HPC / VPN.
def _ensure_hub_timeout() -> None:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")


def _hub_download_clip_weights(
    hf_repo: str,
    hf_file: str,
    cache_dir: str | None,
    *,
    force_download: bool,
) -> str:
    """Download from HF; retry once with force_download if cache has a corrupt partial file."""
    import warnings

    from huggingface_hub import hf_hub_download

    kwargs = {
        "repo_id": hf_repo,
        "filename": hf_file,
        "cache_dir": cache_dir,
        "force_download": force_download,
    }
    try:
        return hf_hub_download(**kwargs)
    except OSError as e:
        msg = str(e)
        if not force_download and (
            "Consistency check failed" in msg or "file should be of size" in msg
        ):
            warnings.warn(
                "HF cache has a partial/corrupt cliprt-oxe-pretrained.pt; "
                "re-downloading with force_download=True.",
                UserWarning,
                stacklevel=2,
            )
            kwargs["force_download"] = True
            return hf_hub_download(**kwargs)
        raise


def _resolve_clip_rt_weights(
    local_weights: Path | None,
    hf_repo: str,
    hf_file: str,
    cache_dir: str | None,
    *,
    force_download: bool = False,
) -> str:
    """Use a local .pt if provided; otherwise download from the hub (long timeout)."""
    if local_weights is not None:
        p = Path(local_weights).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"CLIP-RT weights not found: {p}")
        return str(p)
    _ensure_hub_timeout()
    return _hub_download_clip_weights(
        hf_repo, hf_file, cache_dir, force_download=force_download
    )


def run_clip_rt(
    manifest: Path,
    split: str,
    output: Path,
    hf_repo: str,
    hf_file: str,
    device: str | None,
    cam: str,
    instruction_placeholder: str,
    local_weights: Path | None = None,
    force_download: bool = False,
    *,
    dual_cam: bool | None = None,
) -> dict:
    import numpy as np
    import open_clip
    import torch

    if dual_cam is None:
        dual_cam = env_frozen_dual_cam()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device)

    weight_path = _resolve_clip_rt_weights(
        local_weights,
        hf_repo,
        hf_file,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        force_download=force_download,
    )

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=CLIP_RT_ARCH,
        pretrained=weight_path,
    )
    model = model.to(torch_device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(CLIP_RT_ARCH)

    prompt = (
        "what motion should the robot arm perform to complete the instruction '{}'"
    )
    # Fixed generic intent (manifest has no per-clip language for baselines)
    filled = instruction_placeholder

    commands = load_sdata_command_strings()
    rows = filter_split(load_manifest_rows(manifest), split)

    predictions = []
    y_true: list[int] = []
    y_pred: list[int] = []

    inst_text = prompt.format(filled)
    inst = tokenizer([inst_text]).to(torch_device)
    actions_tok = tokenizer(commands).to(torch_device)

    with torch.no_grad():
        # Precompute text embeddings once (same for all clips)
        inst_features_ref = model.encode_text(inst)
        action_features_ref = model.encode_text(actions_tok)
        inst_features_ref /= inst_features_ref.norm(dim=-1, keepdim=True)
        action_features_ref /= action_features_ref.norm(dim=-1, keepdim=True)

    def _probs_from_pil(pil) -> "np.ndarray":
        image = preprocess(pil).unsqueeze(0).to(torch_device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            context_features = image_features + inst_features_ref
            context_features /= context_features.norm(dim=-1, keepdim=True)
            logits = context_features @ action_features_ref.T
            action_probs = logits.sigmoid()
        return action_probs.squeeze(0).float().cpu().numpy()

    for row in rows:
        gt = row["label"]
        if dual_cam:
            p1 = pil_from_video_mid_frame(row["cam1"])
            p2 = pil_from_video_mid_frame(row["cam2"])
            probs = (_probs_from_pil(p1) + _probs_from_pil(p2)) / 2.0
            cam_note = "cam1+cam2_avg"
        else:
            vpath = row["cam1"] if cam == "cam1" else row["cam2"]
            probs = _probs_from_pil(pil_from_video_mid_frame(vpath))
            cam_note = cam

        pred_idx = int(np.argmax(probs))

        y_true.append(gt)
        y_pred.append(pred_idx)

        predictions.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "label": gt,
                "pred_idx": pred_idx,
                "pred_command": commands[pred_idx],
                "action_probs": probs.tolist(),
                "cam": cam_note,
            }
        )

    metrics = metrics_from_preds(y_true, y_pred, num_classes=len(commands))
    ckpt_ref = (
        f"local:{weight_path}"
        if local_weights is not None
        else f"hf:{hf_repo}/{hf_file}"
    )
    out = {
        "model": "clip_rt",
        "checkpoint": ckpt_ref,
        "frozen": True,
        "split": split,
        "manifest": str(manifest),
        "metrics": metrics,
        "predictions": predictions,
        "dual_cam": dual_cam,
        "note": (
            "Native CLIP-RT contrastive scoring over the eight SData command strings (paper-style primitive selection). "
            + (
                "Averaged action_probs over mid-frames from cam1 and cam2 (aligned with MMFuse two-view input). "
                if dual_cam
                else f"Single camera: {cam}. Set SDATA_FROZEN_DUAL_CAM=1 to average cam1+cam2."
            )
        ),
    }
    write_json(output, out)
    return out


def main():
    p = argparse.ArgumentParser(description="Frozen CLIP-RT (HF weights + open_clip).")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--hf-repo", default=os.environ.get("CLIP_RT_HF_REPO", DEFAULT_HF_REPO))
    p.add_argument("--hf-file", default=os.environ.get("CLIP_RT_HF_FILE", DEFAULT_HF_FILE))
    p.add_argument("--device", default=None)
    p.add_argument("--cam", choices=("cam1", "cam2"), default="cam1")
    p.add_argument(
        "--dual-cam",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Average CLIP-RT scores over cam1+cam2 mid-frames (default: env SDATA_FROZEN_DUAL_CAM or true).",
    )
    p.add_argument(
        "--instruction",
        default=os.environ.get(
            "CLIP_RT_INSTRUCTION", "the therapist's next massage command"
        ),
        help="Fills the paper prompt template (one fixed string for all clips).",
    )
    p.add_argument(
        "--local-weights",
        type=Path,
        default=None,
        help="Path to cliprt-oxe-pretrained.pt (skip hub download; use if HPC blocks or times out).",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore HF cache and re-download the full checkpoint (fixes corrupt partial downloads).",
    )
    args = p.parse_args()
    lw = args.local_weights or (
        Path(p) if (p := os.environ.get("CLIP_RT_LOCAL_WEIGHTS", "").strip()) else None
    )
    fd = args.force_download or (
        os.environ.get("CLIP_RT_FORCE_DOWNLOAD", "").strip() in ("1", "true", "yes")
    )
    run_clip_rt(
        args.manifest,
        args.split,
        args.output,
        args.hf_repo,
        args.hf_file,
        args.device,
        args.cam,
        args.instruction,
        local_weights=lw,
        force_download=fd,
        dual_cam=args.dual_cam,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
