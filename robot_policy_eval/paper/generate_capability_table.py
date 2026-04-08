"""
Generate a Markdown (and JSON) table grouped by **evaluation axis** — same numbers as
`generate_tables.py`, clearer framing for capability-based comparison.

Does not mix different metrics in a single row block without axis labels; each section
states which axis (task success / precision / generalization / robustness) it belongs to.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robot_policy_eval.paper.generate_tables import (
    DEFAULT_POLICY_COLUMNS,
    build_row_values,
)

# (section title, list of (row label, row key in build_row_values output))
AXIS_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Task success & safety",
        [
            ("Task Success Rate (%)", "task_success"),
            ("Safety violations (↓) (count)", "safety_violations"),
        ],
    ),
    (
        "Precision (trajectory & force)",
        [
            ("Trajectory Error (↓) (m)", "trajectory_error"),
            ("Smoothness — mean jerk (↓)", "smoothness_jerk"),
            ("Force deviation (↓) (N)", "force_deviation"),
        ],
    ),
    (
        "Generalization",
        [
            ("Generalization score (%)", "generalization"),
        ],
    ),
    (
        "Robustness",
        [
            ("Robustness — noise test (%)", "robustness_noise"),
        ],
    ),
]


def render_capability_markdown(
    suite: dict[str, Any],
    columns: list[tuple[str, str]] | None = None,
) -> str:
    data = build_row_values(suite, columns)
    lines = [
        "# Table: Capability axes (same metrics as comparative table)",
        "",
        "*Auto-generated — do not edit numbers by hand. Interpretation: see "
        "[`COMPARISON_FRAMEWORK.md`](COMPARISON_FRAMEWORK.md) — compare trade-offs, not raw "
        "parity with unrelated VLA training objectives.*",
        "",
    ]

    for section_title, row_specs in AXIS_SECTIONS:
        lines.append(f"## {section_title}")
        lines.append("")
        hdr = "| Metric | " + " | ".join(data["columns"]) + " |"
        sep = "|" + "|".join(["---"] * (len(data["columns"]) + 1)) + "|"
        lines.append(hdr)
        lines.append(sep)
        for label, rk in row_specs:
            row = data["rows"][rk]
            cells = " | ".join(row["formatted"])
            lines.append(f"| {label} | {cells} |")
        lines.append("")

    lines.append("**Legend:** ↓ = lower is better where indicated. Bold = best in row (within the full comparative table logic).")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_capability_json(
    suite: dict[str, Any],
    columns: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Structured mirror: axis -> rows -> same payload as `build_row_values` rows."""
    flat = build_row_values(suite, columns)
    out: dict[str, Any] = {
        "columns": flat["columns"],
        "policies": flat["policies"],
        "axes": {},
    }
    for section_title, row_specs in AXIS_SECTIONS:
        key = section_title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("&", "and")
        out["axes"][key] = {
            "title": section_title,
            "rows": {rk: flat["rows"][rk] for _, rk in row_specs},
        }
    return out


def generate_capability_from_suite(
    suite: dict[str, Any],
    out_dir: Path | str,
    columns: list[tuple[str, str]] | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_capability_markdown(suite, columns)
    structured = render_capability_json(suite, columns)
    p_md = out_dir / "TABLE_CAPABILITY.md"
    p_json = out_dir / "metrics_capability.json"
    p_md.write_text(md)
    p_json.write_text(json.dumps(structured, indent=2))
    return {"markdown": p_md, "json": p_json}


def generate_capability_from_json(
    metrics_path: Path | str,
    out_dir: Path | str | None = None,
) -> dict[str, Path]:
    metrics_path = Path(metrics_path)
    with open(metrics_path) as f:
        suite = json.load(f)
    out_dir = out_dir or metrics_path.parent
    return generate_capability_from_suite(suite, out_dir)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Generate axis-grouped capability table from metrics_full.json",
    )
    ap.add_argument("metrics_json", type=Path, help="Path to metrics_full.json")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: same as JSON)")
    args = ap.parse_args()
    paths = generate_capability_from_json(args.metrics_json, args.out_dir)
    print("Wrote:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
