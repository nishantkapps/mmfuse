"""
Generate comparative evaluation Markdown / LaTeX tables from `run_evaluation_suite` output.
No hardcoded metric values — all cells derived from `suite` dict (or `metrics_full.json`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Default policy internal names -> manuscript column order (left to right)
DEFAULT_POLICY_COLUMNS: list[tuple[str, str]] = [
    ("dummy_physio", "Your Model (Physio Policy)"),
    ("dummy_vla", "VLA Model"),
    ("hybrid_vla_physio", "Hybrid (VLA + Yours)"),
]


def _sum_safety_violations_normal(suite: dict[str, Any], policy_name: str) -> int:
    total = 0
    for row in suite.get("per_episode", []):
        if row.get("policy") == policy_name and row.get("condition") == "normal":
            total += int(row.get("safety_violations_pred", 0))
    return total


def _generalization_score_percent(suite: dict[str, Any], policy_name: str) -> float:
    """
    Mean over subjects of (episode success rate for that subject), under `normal` only.
    Rewards consistent success across held-out subjects.
    """
    by_subj: dict[str, list[bool]] = {}
    for row in suite.get("per_episode", []):
        if row.get("policy") != policy_name or row.get("condition") != "normal":
            continue
        sid = str(row.get("subject_id", ""))
        by_subj.setdefault(sid, []).append(bool(row.get("success", False)))
    if not by_subj:
        return 0.0
    subj_rates = [float(np.mean(v)) for v in by_subj.values()]
    return 100.0 * float(np.mean(subj_rates))


def _robustness_noise_mean_percent(suite: dict[str, Any], policy_name: str) -> float:
    """Mean task success rate (%) across all `pos_noise_*` conditions."""
    ag = suite.get("aggregates", {}).get(policy_name, {})
    rates: list[float] = []
    for k, v in ag.items():
        if not k.startswith("pos_noise_"):
            continue
        sr = v.get("success_rate", v.get("success", 0.0))
        rates.append(float(sr))
    if not rates:
        return float(ag.get("normal", {}).get("success_rate", 0.0)) * 100.0
    return 100.0 * float(np.mean(rates))


def _column_metrics(suite: dict[str, Any], policy_name: str) -> dict[str, float]:
    ag = suite.get("aggregates", {}).get(policy_name, {})
    if "normal" not in ag:
        raise KeyError(f"No aggregates for policy {policy_name!r}")
    n = ag["normal"]
    return {
        "task_success_pct": 100.0 * float(n.get("success_rate", n.get("success", 0.0))),
        "trajectory_error_m": float(n["mean_l2_position_error"]),
        "jerk": float(n["mean_jerk_pred"]),
        "force_deviation_n": float(n["mean_abs_force_error"]),
        "generalization_pct": _generalization_score_percent(suite, policy_name),
        "robustness_noise_pct": _robustness_noise_mean_percent(suite, policy_name),
        "safety_violations": float(_sum_safety_violations_normal(suite, policy_name)),
    }


def _bold_mask_n(vals: list[float], *, lower_is_better: bool) -> list[bool]:
    """Which columns are best (ties all get bold)."""
    n = len(vals)
    if n < 1 or any(np.isnan(v) for v in vals):
        return [False] * n
    arr = np.array(vals, dtype=np.float64)
    if lower_is_better:
        best = np.min(arr)
        return [abs(v - best) <= 1e-9 * max(1.0, abs(best)) for v in vals]
    best = np.max(arr)
    return [abs(v - best) <= 1e-9 * max(1.0, abs(best)) for v in vals]


def build_row_values(suite: dict[str, Any], columns: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    columns = columns or DEFAULT_POLICY_COLUMNS
    policies = [p for p, _ in columns]
    cols = [_column_metrics(suite, pname) for pname in policies if pname in suite.get("aggregates", {})]
    policy_keys_present = [p for p, _ in columns if p in suite.get("aggregates", {})]
    if len(policy_keys_present) != len(columns):
        missing = [p for p, _ in columns if p not in suite.get("aggregates", {})]
        raise ValueError(f"Suite missing aggregates for policies: {missing}")

    out: dict[str, Any] = {"columns": [lbl for _, lbl in columns], "policies": policy_keys_present, "rows": {}}

    # Row definitions: key, field in col dict, lower_is_better, format string
    specs = [
        ("task_success", "task_success_pct", False, "{:.1f}"),
        ("trajectory_error", "trajectory_error_m", True, "{:.3f}"),
        ("smoothness_jerk", "jerk", True, "{:.3f}"),
        ("force_deviation", "force_deviation_n", True, "{:.2f}"),
        ("generalization", "generalization_pct", False, "{:.1f}"),
        ("robustness_noise", "robustness_noise_pct", False, "{:.1f}"),
        ("safety_violations", "safety_violations", True, "{:.0f}"),
    ]

    metrics_list = [_column_metrics(suite, p) for p in policy_keys_present]

    for row_key, field, lower_better, fmt in specs:
        raw = [float(m[field]) for m in metrics_list]
        bold = _bold_mask_n(raw, lower_is_better=lower_better)
        formatted = []
        for i, v in enumerate(raw):
            s = fmt.format(v)
            if bold[i]:
                s = f"**{s}**"
            formatted.append(s)
        out["rows"][row_key] = {
            "values": [float(v) for v in raw],
            "formatted": formatted,
            "bold": [bool(b) for b in bold],
            "field": field,
        }

    return out


def render_markdown_table(
    suite: dict[str, Any],
    columns: list[tuple[str, str]] | None = None,
) -> str:
    columns = columns or DEFAULT_POLICY_COLUMNS
    data = build_row_values(suite, columns)
    hdr = "| Metric | " + " | ".join(data["columns"]) + " |\n"
    sep = "|" + "|".join(["---"] * (len(data["columns"]) + 1)) + "|\n"

    labels = [
        ("Task Success Rate (%)", "task_success"),
        ("Trajectory Error (↓) (m)", "trajectory_error"),
        ("Smoothness — mean jerk (↓)", "smoothness_jerk"),
        ("Force deviation (↓) (N)", "force_deviation"),
        ("Generalization score (%)", "generalization"),
        ("Robustness — noise test (%)", "robustness_noise"),
        ("Safety violations (↓) (count)", "safety_violations"),
    ]

    lines = [
        "# Table: Comparative Evaluation of Physio Policy vs VLA Models",
        "",
        "*Auto-generated — do not edit numbers by hand; regenerate with `python -m robot_policy_eval.paper.generate_tables` or run the full eval.*",
        "",
        hdr.strip(),
        sep.strip(),
    ]
    for label, rk in labels:
        row = data["rows"][rk]
        cells = " | ".join(row["formatted"])
        lines.append(f"| {label} | {cells} |")
    lines.extend(
        [
            "",
            "**Legend:** ↓ = lower is better. Bold = best in row.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_latex_table(suite: dict[str, Any], columns: list[tuple[str, str]] | None = None) -> str:
    columns = columns or DEFAULT_POLICY_COLUMNS
    data = build_row_values(suite, columns)
    nc = len(data["columns"])
    tabular_spec = "@{}l" + "c" * nc + "@{}"

    lines = [
        r"% Auto-generated from evaluation JSON — requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Comparative evaluation of physio policy vs.\ VLA vs.\ hybrid ($\downarrow$ = lower is better).}",
        r"  \label{tab:comparative_eval}",
        "  \\begin{tabular}{" + tabular_spec + "}",
        r"    \toprule",
        "    \\textbf{Metric} & " + " & ".join(f"\\textbf{{{c}}}" for c in data["columns"]) + r" \\",
        r"    \midrule",
    ]

    row_specs = [
        ("Task success rate (\\%)", "task_success", False),
        ("Trajectory error (m) $\\downarrow$", "trajectory_error", True),
        ("Mean jerk $\\downarrow$", "smoothness_jerk", True),
        ("Force deviation (N) $\\downarrow$", "force_deviation", True),
        ("Generalization (\\%)", "generalization", False),
        ("Robustness — noise (\\%)", "robustness_noise", False),
        ("Safety violations $\\downarrow$", "safety_violations", True),
    ]

    for label, rk, _ in row_specs:
        rdata = data["rows"][rk]
        parts = []
        for i, v in enumerate(rdata["values"]):
            if rk in ("task_success", "generalization", "robustness_noise"):
                cell = f"{v:.1f}"
            elif rk in ("trajectory_error", "smoothness_jerk"):
                cell = f"{v:.3f}"
            elif rk == "force_deviation":
                cell = f"{v:.2f}"
            elif rk == "safety_violations":
                cell = f"{int(round(v))}"
            else:
                cell = f"{v:.3f}"
            if rdata["bold"][i]:
                cell = f"\\textbf{{{cell}}}"
            parts.append(cell)
        lines.append(f"    {label} & " + " & ".join(parts) + r" \\")

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_from_suite(
    suite: dict[str, Any],
    out_dir: Path | str,
    columns: list[tuple[str, str]] | None = None,
) -> dict[str, Path]:
    """Write TABLE.md, table_comparative_evaluation_generated.tex, and metrics_table.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown_table(suite, columns)
    tex = render_latex_table(suite, columns)
    p_md = out_dir / "TABLE.md"
    p_tex = out_dir / "table_comparative_evaluation_generated.tex"
    p_json = out_dir / "metrics_table.json"
    p_md.write_text(md)
    p_tex.write_text(tex)
    p_json.write_text(json.dumps(build_row_values(suite, columns), indent=2))
    return {"markdown": p_md, "latex": p_tex, "json": p_json}


def generate_from_json(metrics_path: Path | str, out_dir: Path | str | None = None) -> dict[str, Path]:
    metrics_path = Path(metrics_path)
    with open(metrics_path) as f:
        suite = json.load(f)
    out_dir = out_dir or metrics_path.parent
    return generate_from_suite(suite, out_dir)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Generate paper tables from metrics_full.json")
    ap.add_argument("metrics_json", type=Path, help="Path to metrics_full.json")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: same as JSON)")
    args = ap.parse_args()
    paths = generate_from_json(args.metrics_json, args.out_dir)
    print("Wrote:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
