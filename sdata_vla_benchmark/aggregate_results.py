#!/usr/bin/env python3
"""
Merge multiple benchmark JSON outputs into a Markdown table (accuracy, macro-F1).
Rows with no `metrics` (e.g. Octo continuous-only, RDT metadata-only) show em dashes, not NaN.
Missing input paths are skipped with a warning (use --strict to error out).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _name_from_file(p: Path) -> str:
    return p.stem.replace("_", " ")


def _fmt_num(v, *, empty: str = "—") -> str:
    if v is None:
        return empty
    if isinstance(v, float) and math.isnan(v):
        return empty
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return str(v)


def _fmt_n(v, *, empty: str = "—") -> str:
    if v is None or v == "":
        return empty
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Result JSON files (e.g. mmfuse_full.json, openvla.json)",
    )
    ap.add_argument("--markdown", type=Path, help="Write Markdown table here")
    ap.add_argument("--show", action="store_true", help="Print table to stdout")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any --inputs path is missing (default: skip missing with a warning)",
    )
    args = ap.parse_args()

    inputs: list[Path] = []
    for p in args.inputs:
        p = Path(p)
        if not p.is_file():
            msg = f"aggregate_results: skip missing file: {p}"
            if args.strict:
                print(msg, file=sys.stderr)
                sys.exit(1)
            print(f"Warning: {msg}", file=sys.stderr)
            continue
        inputs.append(p)

    if not inputs:
        print("aggregate_results: no input files found (all missing?).", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in inputs:
        with open(p) as f:
            data = json.load(f)
        m = data.get("metrics")
        has_metrics = m is not None and isinstance(m, dict) and len(m) > 0
        if not has_metrics:
            m = {}
        name = data.get("model", _name_from_file(p))
        if data.get("mode"):
            name = f"{name} ({data['mode']})"
        reason = ""
        if not has_metrics:
            if data.get("error"):
                reason = str(data["error"])[:120]
            elif data.get("note"):
                reason = str(data["note"])[:120]
            elif data.get("weights_cache"):
                reason = "metrics not defined (metadata/download only)"
        rows.append(
            {
                "name": name,
                "checkpoint": data.get("checkpoint", ""),
                "n": _fmt_n(m.get("n")) if has_metrics else "—",
                "accuracy": _fmt_num(m.get("accuracy")) if has_metrics else "—",
                "macro_f1": _fmt_num(m.get("macro_f1")) if has_metrics else "—",
                "reason": reason,
                "source": str(p),
            }
        )

    lines = [
        "| Model | n | Accuracy | Macro-F1 | Note |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        note = r["reason"].replace("|", "\\|")
        lines.append(
            f"| {r['name']} | {r['n']} | {r['accuracy']} | {r['macro_f1']} | {note} |"
        )
    text = "\n".join(lines) + "\n"
    foot = (
        "\n_Frozen baselines: no SData fine-tuning. MMFuse: same manifest/split. "
        "Rows with “—” had no classification metrics in the JSON (e.g. Octo continuous actions; RDT download-only)._"
    )
    text += foot

    if args.show:
        print(text)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        with open(args.markdown, "w") as f:
            f.write(text)
        print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
