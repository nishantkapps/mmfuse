#!/usr/bin/env python3
"""
Frozen Octo (JAX): native policy forward — load public Octo checkpoint, run sample_actions (or equivalent).

If only continuous actions are returned, we store the raw vector and set pred_idx / metrics to null with a note
(no auxiliary 8-way head; no training on SData).
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdata_vla_benchmark.frozen.common import (
    filter_split,
    load_manifest_rows,
    load_sdata_command_strings,
    pil_from_video_mid_frame,
    write_json,
)


def run_octo(
    manifest: Path,
    split: str,
    output: Path,
    pretrained: str,
    device: str | None,
    cam: str,
) -> dict:
    rows = filter_split(load_manifest_rows(manifest), split)
    commands = load_sdata_command_strings()
    task_text = os.environ.get(
        "OCTO_TASK_TEXT",
        "Perform the massage command that the therapist intends next.",
    )

    predictions = []
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from octo.model.octo_model import OctoModel  # type: ignore
    except ImportError as e:
        out = {
            "model": "octo",
            "checkpoint": pretrained,
            "frozen": True,
            "split": split,
            "manifest": str(manifest),
            "metrics": None,
            "predictions": [],
            "error": f"Octo import failed: {e}. Install Octo per https://github.com/octo-models/octo",
            "note": "Native policy only; when running, raw actions are stored — no SData fine-tuning.",
        }
        write_json(output, out)
        return out

    try:
        model = OctoModel.load_pretrained(pretrained)
    except Exception as e:
        out = {
            "model": "octo",
            "checkpoint": pretrained,
            "frozen": True,
            "split": split,
            "manifest": str(manifest),
            "metrics": None,
            "predictions": [],
            "error": f"load_pretrained failed: {e}\n{traceback.format_exc()}",
        }
        write_json(output, out)
        return out

    rng = jax.random.PRNGKey(int(os.environ.get("OCTO_SEED", "0")))
    for row in rows:
        pil = pil_from_video_mid_frame(row["cam1"] if cam == "cam1" else row["cam2"])
        img = np.array(pil.resize((256, 256)))
        # Many Octo versions expect a leading batch dimension
        observation = {
            "image_primary": np.expand_dims(img, axis=0),
            "timestep": np.zeros(1, dtype=np.int32),
        }
        task = model.create_tasks(texts=[task_text])
        try:
            rng, key = jax.random.split(rng)
            actions = model.sample_actions(observation, task, rng=key)
            raw = np.asarray(actions)
            raw_list = raw.reshape(-1).tolist()[:32]
        except Exception as e:
            raw_list = []
            err = str(e)
        else:
            err = None

        predictions.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "label": row["label"],
                "raw_action_prefix": raw_list,
                "error_forward": err,
                "pred_idx": None,
                "commands_ref": commands,
            }
        )

    out = {
        "model": "octo",
        "checkpoint": pretrained,
        "frozen": True,
        "split": split,
        "manifest": str(manifest),
        "metrics": None,
        "predictions": predictions,
        "note": "Continuous policy output only; no discrete command from frozen Octo — pred_idx null. Compare actions in appendix or define task-specific discretization in paper.",
    }
    write_json(output, out)
    return out


def main():
    p = argparse.ArgumentParser(description="Frozen Octo eval (native sample_actions).")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument(
        "--pretrained",
        default=os.environ.get("OCTO_PRETRAINED", "hf://rail-berkeley/octo-base"),
    )
    p.add_argument("--device", default=None, help="unused (JAX device)")
    p.add_argument("--cam", choices=("cam1", "cam2"), default="cam1")
    args = p.parse_args()
    run_octo(args.manifest, args.split, args.output, args.pretrained, args.device, args.cam)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
