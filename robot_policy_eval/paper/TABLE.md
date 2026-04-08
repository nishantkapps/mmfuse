# Comparative evaluation table

**Numbers are not edited here.** They are **generated** into this file when you run:

```bash
python -m robot_policy_eval.run_evaluation --output-dir robot_policy_eval/outputs/latest
```

Or regenerate from an existing `metrics_full.json`:

```bash
python -m robot_policy_eval.paper.generate_tables robot_policy_eval/outputs/latest/metrics_full.json
```

Outputs in the same directory:

- `TABLE.md` — this file (overwritten)
- `table_comparative_evaluation_generated.tex` — LaTeX
- `metrics_table.json` — structured values + bold flags

See `generate_tables.py` for metric definitions (generalization = per-subject mean success; robustness = mean success across noise levels).
