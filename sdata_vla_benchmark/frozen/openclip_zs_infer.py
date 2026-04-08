#!/usr/bin/env python3
"""
Frozen zero-shot CLIP (OpenCLIP) — image vs text embedding similarity over the eight SData commands.

Uses public LAION/OpenCLIP weights (no robot finetuning). Same 8-class accuracy protocol as CLIP-RT
but standard CLIP scoring (no CLIP-RT fused instruction pathway).
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

# Presets: (key_suffix -> (model_name, pretrained_tag)) — must match open_clip.list_pretrained()
PRESETS: dict[str, tuple[str, str]] = {
    "b32": ("ViT-B-32", "laion2b_e16"),
    "l14": ("ViT-L-14", "laion2b_s32b_b82k"),
    "b16": ("ViT-B-16", "laion2b_s34b_b88k"),
}


def _ensure_hub_timeout() -> None:
    # open_clip pulls LAION weights via huggingface_hub; default read timeout is 10s (often too short).
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")


def run_openclip_zero_shot(
    manifest: Path,
    split: str,
    output: Path,
    model_name: str,
    pretrained: str,
    device: str | None,
    cam: str,
    *,
    text_template: str | None = None,
    model_key: str = "openclip_zs",
    dual_cam: bool | None = None,
) -> dict:
    import numpy as np
    import open_clip
    import torch

    if dual_cam is None:
        dual_cam = env_frozen_dual_cam()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device)

    tpl = text_template if text_template is not None else os.environ.get(
        "OPENCLIP_ZS_TEXT_TEMPLATE", "{}"
    )
    if "{}" not in tpl:
        tpl = "{}"

    _ensure_hub_timeout()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    model = model.to(torch_device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    commands = load_sdata_command_strings()
    texts = [tpl.format(c) for c in commands]
    rows = filter_split(load_manifest_rows(manifest), split)

    predictions = []
    y_true: list[int] = []
    y_pred: list[int] = []

    text_tok = tokenizer(texts).to(torch_device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tok)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()

    def _probs_from_pil(pil) -> "np.ndarray":
        image = preprocess(pil).unsqueeze(0).to(torch_device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            img_feat = model.encode_image(image)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = logit_scale * (img_feat @ text_feat.T)
        return logits.squeeze(0).float().softmax(dim=-1).cpu().numpy()

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
                "probs": probs.tolist(),
                "cam": cam_note,
            }
        )

    metrics = metrics_from_preds(y_true, y_pred, num_classes=len(commands))
    ckpt = f"{model_name}:{pretrained}"
    out = {
        "model": model_key,
        "checkpoint": ckpt,
        "frozen": True,
        "split": split,
        "manifest": str(manifest),
        "metrics": metrics,
        "predictions": predictions,
        "dual_cam": dual_cam,
        "note": (
            "Zero-shot OpenCLIP: image/text similarity over the eight SData command strings "
            f"(text template: {tpl!r}). "
            + (
                "Averaged softmax probs over cam1+cam2 mid-frames (aligned with MMFuse two-view input)."
                if dual_cam
                else f"Single camera: {cam}. Set SDATA_FROZEN_DUAL_CAM=1 to average cam1+cam2."
            )
        ),
    }
    write_json(output, out)
    return out


def main():
    p = argparse.ArgumentParser(description="Frozen zero-shot OpenCLIP on SData commands.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default=None)
    p.add_argument("--cam", choices=("cam1", "cam2"), default="cam1")
    p.add_argument(
        "--dual-cam",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Average OpenCLIP probs over cam1+cam2 (default: SDATA_FROZEN_DUAL_CAM or on).",
    )
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="b32",
        help="Shortcut for model_name + pretrained (see PRESETS in script).",
    )
    p.add_argument("--model-name", default=None, help="Override preset: OpenCLIP architecture name")
    p.add_argument("--pretrained", default=None, help="Override preset: pretrained tag")
    args = p.parse_args()

    mn, pt = PRESETS[args.preset]
    if args.model_name:
        mn = args.model_name
    if args.pretrained:
        pt = args.pretrained

    run_openclip_zero_shot(
        args.manifest,
        args.split,
        args.output,
        mn,
        pt,
        args.device,
        args.cam,
        model_key=f"openclip_{args.preset}",
        dual_cam=args.dual_cam,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
