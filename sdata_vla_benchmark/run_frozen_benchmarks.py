#!/usr/bin/env python3
"""
Frozen comparison on SData (no fine-tuning on your data).

Default suite (**4 rows** with real **8-class** metrics): MMFuse (vision-only + full), CLIP-RT, OpenCLIP B-32.

Optional (no comparable discrete metric in-repo): **rt1**, **rt2**, **saycan**, **openclip_l14**, **openclip_b16**, **openvla**, **octo** — see `--models` / `--help`.

Outputs one JSON per model under --output-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# run_frozen_benchmarks.py lives in sdata_vla_benchmark/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    p = argparse.ArgumentParser(description="Run SData frozen VLA + MMFuse benchmarks.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output-dir", type=Path, default=Path("sdata_vla_benchmark/outputs"))
    p.add_argument(
        "--models",
        nargs="+",
        default=[
            "mmfuse_vision_only",
            "mmfuse_full",
            "clip_rt",
            "openclip_b32",
        ],
        help=(
            "Benchmark keys (default: 4). Add rt1 rt2 saycan (metadata/null metrics), openclip_l14, openvla, octo."
        ),
    )
    p.add_argument("--mmfuse-checkpoint", type=Path, default=None, help="Required for mmfuse_* models")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--openvla-model-id",
        default=None,
        help="HF model id for the 'openvla' slot (any Vision2Seq VLA). Env: OPEN_VLA_MODEL_ID or HF_VLA_MODEL_ID",
    )
    p.add_argument(
        "--hf-vla-report-as",
        default=None,
        help="JSON/table name for that model (env HF_VLA_REPORT_AS). Default: openvla if id contains openvla, else slug from id",
    )
    p.add_argument("--octo-pretrained", default=None, help="Octo pretrained string, default OCTO_PRETRAINED")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    from sdata_vla_benchmark.mmfuse_eval.run_eval import evaluate_mmfuse
    from sdata_vla_benchmark.frozen.hf_vla_core import default_report_name, run_hf_vision2seq_vla
    from sdata_vla_benchmark.frozen.octo_infer import run_octo
    from sdata_vla_benchmark.frozen.rt_infer import run_rt
    from sdata_vla_benchmark.frozen.saycan_infer import run_saycan
    from sdata_vla_benchmark.frozen.clip_rt_infer import run_clip_rt
    from sdata_vla_benchmark.frozen.openclip_zs_infer import PRESETS as OPENCLIP_PRESETS
    from sdata_vla_benchmark.frozen.openclip_zs_infer import run_openclip_zero_shot

    for name in args.models:
        out_path = args.output_dir / f"{name}.json"
        try:
            if name == "mmfuse_vision_only":
                if not args.mmfuse_checkpoint:
                    raise ValueError("--mmfuse-checkpoint required for mmfuse_*")
                evaluate_mmfuse(
                    args.manifest,
                    args.mmfuse_checkpoint,
                    "vision_only",
                    args.split,
                    out_path,
                    device=args.device,
                )
            elif name == "mmfuse_full":
                if not args.mmfuse_checkpoint:
                    raise ValueError("--mmfuse-checkpoint required for mmfuse_*")
                evaluate_mmfuse(
                    args.manifest,
                    args.mmfuse_checkpoint,
                    "full",
                    args.split,
                    out_path,
                    device=args.device,
                )
            elif name == "openvla":
                mid = args.openvla_model_id or os.environ.get(
                    "OPEN_VLA_MODEL_ID", os.environ.get("HF_VLA_MODEL_ID", "openvla/openvla-7b")
                )
                ra = (args.hf_vla_report_as or os.environ.get("HF_VLA_REPORT_AS", "") or "").strip()
                if ra:
                    lbl = ra
                elif "openvla" in mid.lower():
                    lbl = "openvla"
                else:
                    lbl = default_report_name(mid)
                run_hf_vision2seq_vla(
                    args.manifest,
                    args.split,
                    out_path,
                    mid,
                    args.device,
                    "cam1",
                    model_label=lbl,
                )
            elif name == "octo":
                pre = args.octo_pretrained or os.environ.get("OCTO_PRETRAINED", "hf://rail-berkeley/octo-base")
                run_octo(args.manifest, args.split, out_path, pre, args.device, "cam1")
            elif name == "rt1":
                run_rt(args.manifest, args.split, out_path, "rt1")
            elif name == "rt2":
                run_rt(args.manifest, args.split, out_path, "rt2")
            elif name == "saycan":
                run_saycan(args.manifest, args.split, out_path)
            elif name == "clip_rt":
                _lw = os.environ.get("CLIP_RT_LOCAL_WEIGHTS", "").strip()
                _fd = os.environ.get("CLIP_RT_FORCE_DOWNLOAD", "").strip() in (
                    "1",
                    "true",
                    "yes",
                )
                run_clip_rt(
                    args.manifest,
                    args.split,
                    out_path,
                    os.environ.get("CLIP_RT_HF_REPO", "clip-rt/clip-rt-oxe-pretrained"),
                    os.environ.get("CLIP_RT_HF_FILE", "cliprt-oxe-pretrained.pt"),
                    args.device,
                    "cam1",
                    os.environ.get(
                        "CLIP_RT_INSTRUCTION", "the therapist's next massage command"
                    ),
                    local_weights=Path(_lw) if _lw else None,
                    force_download=_fd,
                )
            elif name.startswith("openclip_"):
                key = name.removeprefix("openclip_")
                if key not in OPENCLIP_PRESETS:
                    raise ValueError(
                        f"Unknown OpenCLIP preset {name!r}; choose one of "
                        f"{['openclip_' + k for k in OPENCLIP_PRESETS]}"
                    )
                mn, pt = OPENCLIP_PRESETS[key]
                run_openclip_zero_shot(
                    args.manifest,
                    args.split,
                    out_path,
                    mn,
                    pt,
                    args.device,
                    "cam1",
                    model_key=name,
                )
            else:
                raise ValueError(f"Unknown model key: {name}")
            with open(out_path) as f:
                payload = json.load(f)
            if payload.get("metrics") is not None:
                summary[name] = payload["metrics"]
            elif payload.get("error"):
                summary[name] = {"error": payload["error"]}
            else:
                summary[name] = payload
        except Exception as e:
            summary[name] = {"error": str(e)}
            err_path = args.output_dir / f"{name}_error.txt"
            err_path.write_text(str(e))

    print(json.dumps({"output_dir": str(args.output_dir), "summary_metrics": summary}, indent=2))


if __name__ == "__main__":
    main()
