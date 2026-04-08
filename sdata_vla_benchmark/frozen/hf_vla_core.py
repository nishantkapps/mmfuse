"""
Shared frozen eval: Hugging Face models that load via AutoModelForVision2Seq + generate (image + text).

Same manifest, same 8 SData command strings, same post-hoc text→label for metrics.
No training on SData.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdata_vla_benchmark.frozen.common import (
    filter_split,
    load_manifest_rows,
    load_sdata_command_strings,
    metrics_from_preds,
    pil_from_video_mid_frame,
    posthoc_text_to_label,
    write_json,
)


def build_prompt(commands: list[str]) -> str:
    lines = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(commands))
    return (
        "You view a single camera image from a therapeutic massage session. "
        "Exactly one of the following commands applies as the intended next instruction.\n"
        f"{lines}\n"
        "Reply with only the exact command phrase (verbatim from the list) that best matches the scene."
    )


def _ensure_attn_implementation_on_config(config) -> None:
    """
    OpenVLA's remote Prismatic code targets transformers>=4.40, which sets
    `_attn_implementation` on configs. Older transformers omit it → AttributeError
    in modeling_prismatic. Patch missing fields so frozen eval works without
    upgrading the shared environment.
    """
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
        for key in (
            "text_config",
            "vision_config",
            "llm_config",
            "encoder_config",
            "decoder_config",
        ):
            sub = getattr(cfg, key, None)
            if sub is not None:
                stack.append(sub)


def default_report_name(model_id: str) -> str:
    """Short safe name for JSON / tables."""
    slug = model_id.rstrip("/").split("/")[-1]
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", slug)[:80]
    return slug or "hf_vla"


def run_hf_vision2seq_vla(
    manifest: Path,
    split: str,
    output: Path,
    model_id: str,
    device: str | None,
    cam: str,
    model_label: str | None = None,
) -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    rows = filter_split(load_manifest_rows(manifest), split)
    commands = load_sdata_command_strings()
    prompt = build_prompt(commands)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    _ensure_attn_implementation_on_config(model_config)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        config=model_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model.eval()
    model.to(device)

    label_out = model_label or default_report_name(model_id)

    predictions = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for row in rows:
        gt = row["label"]
        vpath = row["cam1"] if cam == "cam1" else row["cam2"]
        image = pil_from_video_mid_frame(vpath)

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {k: v.to(device) for k, v in dict(inputs).items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=int(os.environ.get("HF_VLA_MAX_NEW_TOKENS", "128")),
                do_sample=False,
                num_beams=1,
            )

        new_tokens = gen_ids[0, input_len:]
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            raw = tok.decode(new_tokens, skip_special_tokens=True)
        else:
            raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        pred_idx, match_method = posthoc_text_to_label(raw, commands)
        pred_phrase = commands[pred_idx]

        y_true.append(gt)
        y_pred.append(pred_idx)

        predictions.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "label": gt,
                "raw_output": raw,
                "pred_idx": pred_idx,
                "pred_command": pred_phrase,
                "posthoc_match": match_method,
                "cam_used": cam,
            }
        )

    metrics = metrics_from_preds(y_true, y_pred, num_classes=len(commands))
    out = {
        "model": label_out,
        "checkpoint": model_id,
        "frozen": True,
        "split": split,
        "manifest": str(manifest),
        "metrics": metrics,
        "predictions": predictions,
        "note": (
            "AutoModelForVision2Seq.generate(); pred_idx from posthoc_text_to_label on decoded text. "
            "Swap --model-id for any HF VLA that supports this API."
        ),
    }
    write_json(output, out)
    return out
