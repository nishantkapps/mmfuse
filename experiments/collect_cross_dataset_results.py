#!/usr/bin/env python3
"""
Collect results from all cross-dataset experiments into a consolidated table.
Run after experiments/run_all_experiments.py.

Primary output (paper-ready, one row, datasets as columns):
- experiments/results_table.csv   - Single table: Model | L1 | L2 | L3 | A1-A8 | EgoSchema | NeXTQA | VideoMME | ADL-X | Avg
- experiments/results_table.tex   - LaTeX table ready for \\input{} in paper

Additional outputs:
- cross_dataset_summary.csv, cross_dataset_per_class.csv
- cross_dataset_summary.tex, cross_dataset_summary.html
- figures/cross_dataset_accuracy.png

Usage:
  python experiments/collect_cross_dataset_results.py
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=None, help="Results root (default: experiments/results)")
    p.add_argument("--output", default=None, help="Output path (default: experiments/cross_dataset_summary)")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent
    results_dir = Path(args.results_dir) if args.results_dir else script_dir / "results"
    out_path = Path(args.output) if args.output else script_dir / "cross_dataset_summary"

    config_path = script_dir / "config" / "datasets.yaml"
    with open(config_path) as f:
        configs = yaml.safe_load(f)

    sdata_classes = configs.pop("SDATA_CLASSES", [
        "Start the Massage", "Focus Here", "Move down a little bit", "Go Back Up",
        "Stop. Pause for a second", "Move to the Left", "Move to the Right", "Right there, perfect spot",
    ])
    viscop_levels = configs.pop("VISCOP_LEVELS", [
        "L1 (Object Placement)", "L2 (Novel Combination)", "L3 (Novel Object)",
    ])

    rows = []
    for name, cfg in configs.items():
        results_file = results_dir / name / "results.json"
        if not results_file.exists():
            rows.append({
                "dataset": cfg["name"],
                "key": name,
                "accuracy": None,
                "num_samples": None,
                "movement_mse": None,
                "has_movement_head": cfg.get("has_movement_head", False),
                "per_class": None,
                "class_names": cfg.get("class_names"),
                "taxonomy": cfg.get("taxonomy", "native"),
                "per_level": None,
                "status": "not run",
            })
            continue
        with open(results_file) as f:
            data = json.load(f)
        rows.append({
            "dataset": data.get("dataset", cfg["name"]),
            "key": name,
            "accuracy": data.get("accuracy"),
            "num_samples": data.get("num_samples"),
            "movement_mse": data.get("movement_mse"),
            "has_movement_head": data.get("has_movement_head", cfg.get("has_movement_head", False)),
            "per_class": data.get("per_class"),
            "class_names": cfg.get("class_names"),
            "taxonomy": cfg.get("taxonomy", "native"),
            "per_level": data.get("per_level"),  # VisCoP L1/L2/L3
            "status": "ok",
        })

    # CSV (overall + per-class)
    csv_path = Path(str(out_path) + ".csv") if "." not in str(out_path) else out_path
    if not str(csv_path).endswith(".csv"):
        csv_path = script_dir / "cross_dataset_summary.csv"
    with open(csv_path, "w") as f:
        f.write("dataset,accuracy,num_samples,movement_mse,has_movement_head,status\n")
        for r in rows:
            acc = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else ""
            n = r["num_samples"] or ""
            mse = f"{r['movement_mse']:.4f}" if r["movement_mse"] is not None else ""
            mov = "yes" if r["has_movement_head"] else "no"
            f.write(f'"{r["dataset"]}",{acc},{n},{mse},{mov},{r["status"]}\n')
    print(f"CSV saved to {csv_path}")

    # Per-class CSV
    pc_path = script_dir / "cross_dataset_per_class.csv"
    with open(pc_path, "w") as f:
        f.write("dataset,class_name,accuracy,precision,recall,f1,support\n")
        for r in rows:
            if not r.get("per_class") or not r.get("class_names"):
                continue
            for i, cname in enumerate(r["class_names"]):
                key = str(i)
                if key in r["per_class"]:
                    m = r["per_class"][key]
                    acc_pc = m.get("accuracy") if m.get("accuracy") is not None else ""
                    prec = m.get("precision") if m.get("precision") is not None else ""
                    rec = m.get("recall") if m.get("recall") is not None else ""
                    f1 = m.get("f1-score") if m.get("f1-score") is not None else ""
                    sup = m.get("support", "")
                    cname_esc = str(cname).replace('"', '""')
                    f.write(f'"{r["dataset"]}","{cname_esc}",{acc_pc},{prec},{rec},{f1},{sup}\n')
    print(f"Per-class CSV saved to {pc_path}")

    # Build results by dataset key for unified table
    by_key = {r["key"]: r for r in rows}
    vb = by_key.get("vima_bench", {})
    pc_vb = vb.get("per_class") or {}
    vima_acc = vb.get("accuracy")
    sd = by_key.get("sdata", {})
    pc_sd = sd.get("per_class") or {}
    human_accs = []
    for k in ["egoschema", "nextqa", "video_mme", "charades"]:
        acc = by_key.get(k, {}).get("accuracy")
        if acc is not None:
            human_accs.append(acc)
    avg = (sum(human_accs) / len(human_accs) * 100) if human_accs else "---"

    # Build one-row unified table (datasets as columns)
    col_headers = ["Model", "L1", "L2", "L3"] + [f"A{i+1}" for i in range(8)] + [
        "EgoSchema", "NeXTQA", "VideoMME", "ADL-X", "Avg"
    ]
    vals = ["MMFuse"]
    for i in range(3):
        v = pc_vb.get(str(i), {}).get("accuracy")
        vals.append(f"{100*v:.1f}" if v is not None else "---")
    for i in range(8):
        v = pc_sd.get(str(i), {}).get("accuracy")
        vals.append(f"{100*v:.1f}" if v is not None else "---")
    for k in ["egoschema", "nextqa", "video_mme", "charades"]:
        acc = by_key.get(k, {}).get("accuracy")
        vals.append(f"{100*acc:.1f}" if acc is not None else "---")
    vals.append(f"{avg:.1f}" if isinstance(avg, (int, float)) else str(avg))

    # --- PRIMARY OUTPUT: Single consolidated table (paper-ready, one row) ---
    table_csv = script_dir / "results_table.csv"
    with open(table_csv, "w") as f:
        f.write(",".join(col_headers) + "\n")
        f.write(",".join(str(x) for x in vals) + "\n")
    print(f"Consolidated table: {table_csv}")

    table_tex = script_dir / "results_table.tex"
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{MMFuse cross-dataset results. L1/L2/L3: VIMA-Bench. A1--A8: SData classes. EgoSchema, NeXTQA, VideoMME, ADL-X: video QA benchmarks.}",
        r"\label{tab:mmfuse_results}",
        r"\begin{tabular}{l" + "c" * (len(col_headers) - 1) + "}",
        r"\toprule",
        r"Model & L1 & L2 & L3 & " + " & ".join([f"A{i+1}" for i in range(8)]) + r" & EgoSchema & NeXTQA & VideoMME & ADL-X & Avg \\",
        r"\midrule",
        " & ".join(vals) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(table_tex, "w") as f:
        f.write("\n".join(tex_lines))
    print(f"LaTeX table (paper-ready): {table_tex}")

    # Legacy: cross_dataset_unified.csv (same as results_table)
    unified_path = script_dir / "cross_dataset_unified.csv"
    with open(unified_path, "w") as f:
        f.write(",".join(col_headers) + "\n")
        f.write(",".join(str(x) for x in vals) + "\n")
        f.write("\n# SData: A1=" + sdata_classes[0] + "; A2=" + sdata_classes[1] + "; ... A8=" + sdata_classes[7] + "\n")
    print(f"Unified CSV: {unified_path}")

    # LaTeX tables (VisCoP-style)
    tex_path = Path(str(out_path) + ".tex") if "." not in str(out_path) else out_path
    if not str(tex_path).endswith(".tex"):
        tex_path = script_dir / "cross_dataset_summary.tex"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{MMFuse cross-dataset evaluation (VisCoP-aligned benchmarks). Overall accuracy and per-dataset metrics. Movement head is unique to our model.}",
        r"\label{tab:cross_dataset}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & Acc. (\%) & N & Movement MSE \\",
        r"\midrule",
    ]
    for r in rows:
        acc_str = f"{100*r['accuracy']:.2f}" if r["accuracy"] is not None else "---"
        n_str = str(r["num_samples"]) if r["num_samples"] else "---"
        mse_str = f"{r['movement_mse']:.4f}" if r["movement_mse"] is not None else "---"
        name = r["dataset"].replace("_", r"\_")
        lines.append(f"{name} & {acc_str} & {n_str} & {mse_str} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    # VisCoP-style unified table: L1 | L2 | L3 | A1..A8 | EgoSchema | NeXTQA | VideoMME | ADL-X | Avg
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{MMFuse results (VisCoP-style). L1/L2/L3: VIMA-Bench. A1--A8: SData classes. Human: EgoSchema, NeXTQA, VideoMME, ADL-X.}")
    lines.append(r"\label{tab:cross_dataset_unified}")
    ncol = 1 + 3 + 8 + 4 + 1  # Model + L1/L2/L3 + A1..A8 + 4 datasets + Avg
    lines.append(r"\begin{tabular}{l" + "c" * (ncol - 1) + "}")
    lines.append(r"\toprule")
    a_cols = " & ".join([f"A{i+1}" for i in range(8)])
    lines.append(r"Model & L1 & L2 & L3 & " + a_cols + r" & EgoSchema & NeXTQA & VideoMME & ADL-X & Avg \\")
    lines.append(r"\midrule")
    cells = ["MMFuse"]
    for i in range(3):
        v = pc_vb.get(str(i), {}).get("accuracy")
        cells.append(f"{100*v:.1f}" if v is not None else "---")
    for i in range(8):
        v = pc_sd.get(str(i), {}).get("accuracy")
        cells.append(f"{100*v:.1f}" if v is not None else "---")
    for k in ["egoschema", "nextqa", "video_mme", "charades"]:
        acc = by_key.get(k, {}).get("accuracy")
        cells.append(f"{100*acc:.1f}" if acc is not None else "---")
    cells.append(f"{avg:.1f}" if isinstance(avg, (int, float)) else str(avg))
    lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    # SData class descriptions (text below table)
    lines.append(r"\vspace{0.3em}")
    _esc = r"\_"
    lines.append(r"\footnotesize\textbf{SData:} " + ", ".join([f"A{i+1}={s.replace('_', _esc)}" for i, s in enumerate(sdata_classes)]) + r". \textbf{VisCoP:} L1=Object Placement, L2=Novel Combination, L3=Novel Object.")
    lines.append(r"\end{table}")

    # Per-class table (actions / our 8 classes) for datasets with breakdown
    per_class_rows = [r for r in rows if r.get("per_class") and r.get("class_names")]
    if per_class_rows:
        lines.append("")
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Per-class accuracy. SData 8 actions + VisCoP L1/L2/L3.}")
        lines.append(r"\label{tab:cross_dataset_per_class}")
        lines.append(r"\begin{tabular}{llcccc}")
        lines.append(r"\toprule")
        lines.append(r"Dataset & Class & Acc. (\%) & Prec. & Recall & F1 \\")
        lines.append(r"\midrule")
        for idx, r in enumerate(per_class_rows):
            names = r["class_names"]
            pc = r["per_class"]
            for i, cname in enumerate(names):
                key = str(i)
                if key in pc:
                    m = pc[key]
                    acc_pc = f"{100*m.get('accuracy', 0):.1f}" if m.get("accuracy") is not None else "---"
                    prec = f"{m.get('precision', 0):.2f}" if m.get("precision") is not None else "---"
                    rec = f"{m.get('recall', 0):.2f}" if m.get("recall") is not None else "---"
                    f1 = f"{m.get('f1-score', 0):.2f}" if m.get("f1-score") is not None else "---"
                    ds = r["dataset"].replace("_", r"\_")
                    cls = str(cname).replace("_", r"\_")
                    lines.append(f"{ds} & {cls} & {acc_pc} & {prec} & {rec} & {f1} \\\\")
            if idx < len(per_class_rows) - 1:
                lines.append(r"\midrule")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX saved to {tex_path}")

    # HTML table - Unified table FIRST (L1 | L2 | L3 | A1..A8 | Dataset-specific)
    html_path = script_dir / "cross_dataset_summary.html"
    html_lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>MMFuse Cross-Dataset Results</title>",
        "<style>table{border-collapse:collapse;font-family:sans-serif}th,td{border:1px solid #333;padding:8px 12px;text-align:left}th{background:#1a1a2e;color:#eee}.num{text-align:right}</style>",
        "</head><body><h1>MMFuse Cross-Dataset Results (VisCoP-aligned)</h1>",
        "<h2>Unified Table: L1 | L2 | L3 | A1-A8 | Dataset-specific</h2>",
    ]
    html_lines.append("<table><tr><th>Model</th><th>L1</th><th>L2</th><th>L3</th>")
    for i in range(8):
        html_lines.append(f"<th>A{i+1}</th>")
    html_lines.append("<th>EgoSchema</th><th>NeXTQA</th><th>VideoMME</th><th>ADL-X</th><th>Avg</th></tr>")
    row_cells = ["<tr><td>MMFuse</td>"]
    for i in range(3):
        v = pc_vb.get(str(i), {}).get("accuracy")
        row_cells.append(f"<td class='num'>{100*v:.1f}</td>" if v is not None else "<td>---</td>")
    for i in range(8):
        v = pc_sd.get(str(i), {}).get("accuracy")
        row_cells.append(f"<td class='num'>{100*v:.1f}</td>" if v is not None else "<td>---</td>")
    for k in ["egoschema", "nextqa", "video_mme", "charades"]:
        acc = by_key.get(k, {}).get("accuracy")
        row_cells.append(f"<td class='num'>{100*acc:.1f}</td>" if acc is not None else "<td>---</td>")
    row_cells.append(f"<td class='num'>{avg:.1f}</td>" if isinstance(avg, (int, float)) else "<td>---</td>")
    html_lines.append("".join(row_cells) + "</tr>")
    html_lines.append("</table>")
    html_lines.append("<p><small><b>SData classes:</b> " + "; ".join([f"A{i+1}={c}" for i, c in enumerate(sdata_classes)]) + ". <b>VisCoP:</b> L1=Object Placement, L2=Novel Combination, L3=Novel Object.</small></p>")

    # Per-dataset summary (Dataset, Acc, N, MSE)
    html_lines.append("<h2>Per-Dataset Summary</h2><table><tr><th>Dataset</th><th>Acc (%)</th><th>N</th><th>Movement MSE</th><th>Status</th></tr>")
    for r in rows:
        acc = f"{100*r['accuracy']:.2f}" if r["accuracy"] is not None else "---"
        n = str(r["num_samples"]) if r["num_samples"] else "---"
        mse = f"{r['movement_mse']:.4f}" if r["movement_mse"] is not None else "---"
        html_lines.append(f"<tr><td>{r['dataset']}</td><td class='num'>{acc}</td><td class='num'>{n}</td><td class='num'>{mse}</td><td>{r['status']}</td></tr>")
    html_lines.append("</table>")

    if per_class_rows:
        html_lines.append("<h2>Per-Class (actions / 8 classes)</h2><table><tr><th>Dataset</th><th>Class</th><th>Acc (%)</th><th>Prec</th><th>Recall</th><th>F1</th></tr>")
        for r in per_class_rows:
            for i, cname in enumerate(r["class_names"]):
                key = str(i)
                if key in r["per_class"]:
                    m = r["per_class"][key]
                    acc_pc = f"{100*m.get('accuracy', 0):.1f}" if m.get("accuracy") is not None else "---"
                    prec = f"{m.get('precision', 0):.2f}" if m.get("precision") is not None else "---"
                    rec = f"{m.get('recall', 0):.2f}" if m.get("recall") is not None else "---"
                    f1 = f"{m.get('f1-score', 0):.2f}" if m.get("f1-score") is not None else "---"
                    html_lines.append(f"<tr><td>{r['dataset']}</td><td>{cname}</td><td class='num'>{acc_pc}</td><td class='num'>{prec}</td><td class='num'>{rec}</td><td class='num'>{f1}</td></tr>")
        html_lines.append("</table>")
    html_lines.append("</body></html>")
    with open(html_path, "w") as f:
        f.write("\n".join(html_lines))
    print(f"HTML table saved to {html_path}")

    # Chart images
    if HAS_MATPLOTLIB:
        figs_dir = script_dir / "figures"
        figs_dir.mkdir(exist_ok=True)
        valid_rows = [r for r in rows if r["accuracy"] is not None and r["status"] == "ok"]
        if valid_rows:
            # Bar chart: overall accuracy per dataset
            fig, ax = plt.subplots(figsize=(10, 5))
            names = [r["dataset"] for r in valid_rows]
            accs = [100 * r["accuracy"] for r in valid_rows]
            colors = plt.cm.viridis([a / 100 for a in accs])
            bars = ax.barh(names, accs, color=colors)
            ax.set_xlabel("Accuracy (%)")
            ax.set_title("MMFuse Cross-Dataset Accuracy (VisCoP-aligned)")
            ax.set_xlim(0, 100)
            ax.invert_yaxis()
            for bar, a in zip(bars, accs):
                ax.text(a + 1, bar.get_y() + bar.get_height() / 2, f"{a:.1f}%", va="center", fontsize=9)
            fig.tight_layout()
            fig.savefig(figs_dir / "cross_dataset_accuracy.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Chart saved to {figs_dir / 'cross_dataset_accuracy.png'}")

            # Per-class bar chart (grouped by dataset)
            per_class_rows = [r for r in rows if r.get("per_class") and r.get("class_names")]
            if per_class_rows:
                fig, ax = plt.subplots(figsize=(12, 6))
                x_labels, accs_pc, colors_pc = [], [], []
                palette = plt.cm.Set3.colors
                for idx, r in enumerate(per_class_rows):
                    for i, cname in enumerate(r["class_names"]):
                        key = str(i)
                        if key in r["per_class"]:
                            acc = r["per_class"][key].get("accuracy")
                            if acc is not None:
                                x_labels.append(f"{r['dataset'][:8]}\n{cname}")
                                accs_pc.append(100 * acc)
                                colors_pc.append(palette[idx % len(palette)])
                if x_labels:
                    x = range(len(x_labels))
                    ax.bar(x, accs_pc, color=colors_pc)
                    ax.set_xticks(x)
                    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
                    ax.set_ylabel("Accuracy (%)")
                    ax.set_title("Per-Class Accuracy (actions / 8 classes)")
                    ax.set_ylim(0, 100)
                    fig.tight_layout()
                    fig.savefig(figs_dir / "cross_dataset_per_class.png", dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"Chart saved to {figs_dir / 'cross_dataset_per_class.png'}")
        else:
            print("No results to plot. Run experiments first.")
    else:
        print("matplotlib not available. Skipping charts.")

    # Console: Unified table (L1 | L2 | L3 | A1..A8 | Dataset-specific)
    print(f"\n{'='*100}")
    print("UNIFIED TABLE: L1/L2/L3 | A1-A8 | EgoSchema | NeXTQA | VideoMME | ADL-X | Avg")
    print(f"{'='*100}")
    hdr = "Model   L1    L2    L3   " + "  ".join([f"A{i+1}" for i in range(8)]) + "  EgoSchema NeXTQA VideoMME ADL-X   Avg"
    print(hdr)
    print("-" * len(hdr))
    cells = ["MMFuse"]
    for i in range(3):
        v = pc_vb.get(str(i), {}).get("accuracy")
        cells.append(f"{100*v:.1f}" if v is not None else "---")
    for i in range(8):
        v = pc_sd.get(str(i), {}).get("accuracy")
        cells.append(f"{100*v:.1f}" if v is not None else "---")
    for k in ["egoschema", "nextqa", "video_mme", "charades"]:
        acc = by_key.get(k, {}).get("accuracy")
        cells.append(f"{100*acc:.1f}" if acc is not None else "---")
    cells.append(f"{avg:.1f}" if isinstance(avg, (int, float)) else str(avg))
    print("  ".join(cells))
    print(f"{'='*100}")
    print("SData: A1=Start, A2=Focus, A3=Down, A4=Up, A5=Stop, A6=Left, A7=Right, A8=Perfect")
    print("VisCoP: L1=Object Placement, L2=Novel Combination, L3=Novel Object")
    print(f"{'='*100}")

    # Console: Per-dataset summary (Dataset, Acc, N, MSE)
    print(f"\n{'='*70}")
    print("PER-DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<25} {'Accuracy':<12} {'N':<8} {'Movement MSE':<14} {'Status'}")
    print("-" * 70)
    for r in rows:
        acc = f"{100*r['accuracy']:.2f}%" if r["accuracy"] is not None else "---"
        n = str(r["num_samples"]) if r["num_samples"] else "---"
        mse = f"{r['movement_mse']:.4f}" if r["movement_mse"] is not None else "---"
        print(f"{r['dataset']:<25} {acc:<12} {n:<8} {mse:<14} {r['status']}")
    print(f"{'='*70}")

    # Per-class breakdown (actions / our 8 classes)
    per_class_rows = [r for r in rows if r.get("per_class") and r.get("class_names")]
    if per_class_rows:
        print(f"\n{'='*70}")
        print("PER-CLASS ACCURACY (actions / our 8 classes)")
        print(f"{'='*70}")
        for r in per_class_rows:
            print(f"\n{r['dataset']}:")
            print(f"  {'Class':<12} {'Acc':<10} {'Prec':<8} {'Recall':<8} {'F1':<8}")
            print("  " + "-" * 46)
            for i, cname in enumerate(r["class_names"]):
                key = str(i)
                if key in r["per_class"]:
                    m = r["per_class"][key]
                    acc_pc = f"{100*m.get('accuracy', 0):.1f}%" if m.get("accuracy") is not None else "---"
                    prec = f"{m.get('precision', 0):.2f}" if m.get("precision") is not None else "---"
                    rec = f"{m.get('recall', 0):.2f}" if m.get("recall") is not None else "---"
                    f1 = f"{m.get('f1-score', 0):.2f}" if m.get("f1-score") is not None else "---"
                    print(f"  {cname:<12} {acc_pc:<10} {prec:<8} {rec:<8} {f1:<8}")
        print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
