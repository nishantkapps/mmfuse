#!/usr/bin/env python3
"""
RDT (Robotics Diffusion Transformer) — weights on Hugging Face, native inference from upstream repo.

Google's original RT-1 / RT-2 checkpoints are **not** published as a single `transformers.AutoModel` on HF.
The closest public, hub-hosted **imitation / VLA** line with downloadable weights is **RDT**:
  - robotics-diffusion-transformer/rdt-170m
  - robotics-diffusion-transformer/rdt-1b

This script always **downloads** the snapshot via `huggingface_hub`. Full **diffusion** forward pass
requires the official codebase (policy.step, language embeddings, proprio) — see:
  https://github.com/thu-ml/RoboticsDiffusionTransformer

Set **RDT_ROOT** to a clone of that repository and install its deps to enable native `create_model` / `policy.step`
in a follow-up; until then we record weights path + manifest size so nothing is "fake trained".
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdata_vla_benchmark.frozen.common import filter_split, load_manifest_rows, write_json

# Map legacy names rt1/rt2 to HF RDT sizes
VARIANT_TO_REPO = {
    "rt1": "robotics-diffusion-transformer/rdt-170m",
    "rt2": "robotics-diffusion-transformer/rdt-1b",
    "rdt_170m": "robotics-diffusion-transformer/rdt-170m",
    "rdt_1b": "robotics-diffusion-transformer/rdt-1b",
}


def run_rdt(manifest: Path, split: str, output: Path, variant: str) -> dict:
    rows = filter_split(load_manifest_rows(manifest), split)
    repo_id = VARIANT_TO_REPO.get(variant, VARIANT_TO_REPO["rt1"])

    cache_path = None
    err_dl = None
    try:
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
        from huggingface_hub import snapshot_download

        cache_path = snapshot_download(repo_id=repo_id, cache_dir=os.environ.get("HF_HUB_CACHE"))
    except Exception as e:
        err_dl = str(e)

    rdt_root = os.environ.get("RDT_ROOT", "").strip()

    setup_doc = str(REPO_ROOT / "sdata_vla_benchmark" / "EXTERNAL_BASELINES.md")
    req_rdt = str(REPO_ROOT / "sdata_vla_benchmark" / "requirements-rdt.txt")

    out = {
        "model": variant,
        "proxy_for": "Google RT-1/RT-2 are not on HF as one-line checkpoints; this entry uses RDT weights as the public robotics-transformer baseline.",
        "checkpoint": repo_id,
        "frozen": True,
        "split": split,
        "manifest": str(manifest),
        "weights_cache": cache_path,
        "download_error": err_dl,
        "metrics": None,
        "predictions": [],
        "n_manifest_rows": len(rows),
        "setup_doc": setup_doc,
        "requirements_file": req_rdt,
        "note": (
            "Full RDT prediction: separate conda env, clone thu-ml/RoboticsDiffusionTransformer, "
            "install per requirements-rdt.txt + repo README (torch, flash-attn, SigLIP, precomputed lang embeds, "
            "proprio + 6 images per step). Outputs continuous robot actions — not SData 8-class labels without "
            f"your own mapping. See setup_doc: {setup_doc}"
        ),
    }
    if rdt_root:
        out["RDT_ROOT"] = rdt_root
        out["note"] += f" RDT_ROOT is set ({rdt_root}) — wire create_model/policy.step in this file when ready."

    if err_dl:
        out["error"] = f"snapshot_download failed: {err_dl}"

    write_json(output, out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument(
        "--variant",
        choices=("rt1", "rt2", "rdt_170m", "rdt_1b"),
        default="rdt_170m",
    )
    args = p.parse_args()
    run_rdt(args.manifest, args.split, args.output, args.variant)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
