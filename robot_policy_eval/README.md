# Robot policy evaluation framework

Modular **Python** tooling to compare policies in a **shared trajectory space** (pose + force over time), with metrics suited to **capability-based** comparison — not raw score parity against VLAs trained on unrelated data. Read **[`paper/COMPARISON_FRAMEWORK.md`](paper/COMPARISON_FRAMEWORK.md)** before writing paper text.

**Testing several models on your trajectory dataset:** use **[`cross_model`](cross_model/)** — [`paper/CROSS_MODEL_EVAL.md`](paper/CROSS_MODEL_EVAL.md). CLI: `python -m robot_policy_eval.cross_model --dataset your.json --output-dir ...`. Plug in real VLAs by implementing `Policy` (see [`policies/vla_adapter.py`](policies/vla_adapter.py)); the demo CLI only runs built-in dummies.

**MMFuse + SOTA-style baselines (same trajectory metrics):** install [`requirements-baselines.txt`](requirements-baselines.txt), then:

```bash
python -m robot_policy_eval.sota \
  --dataset robot_policy_eval/data/sdata_trajectories.json \
  --repo-root . \
  --mmfuse-checkpoint checkpoints/your_ckpt.pt \
  --baselines mmfuse clip_rt openclip_b32 hybrid_demo \
  --output-dir robot_policy_eval/outputs/sota
```

Add `openvla`, `openclip_l14`, `rt1_stub`, `saycan_stub`, `dummy_physio`, etc. See [`baselines/factory.py`](baselines/factory.py). Stubs produce valid trajectories for metric computation but are not real robot inference.

Built-in **mock** policies (replace with real `Policy` subclasses):

1. **Task-specific (physio)** — `DummyPhysioPolicy` (bounded tracking noise on GT).
2. **VLA-style** — `DummyVLAPolicy` (coarse / subsampled plan).
3. **Hybrid (recommended differentiator)** — `HybridPolicy`: high-level coarse plan → interpolation → low-level refinement. Instantiate with **your** planner + controller to measure gains vs each branch alone (same metrics on all three).

## Trajectory format

Each timestep: `position` (x,y,z), `orientation` (roll,pitch,yaw), `force`, `timestamp`.

## Metrics

- Mean L2 position error (resampled trajectories)
- DTW on 3D positions
- Mean jerk (from discrete derivatives w.r.t. time)
- Mean absolute force error
- Task success (RMSE threshold + no force violations)
- Safety violation count

## Robustness

Gaussian noise on **input** positions; random timestep **occlusion**. Metrics are computed vs **clean** ground truth.

## Generalization

`TrajectoryDataset.split_by_subjects(train_subjects, test_subjects)` — evaluate only on **unseen** subject IDs.

## Run

**Synthetic demo (no dataset file):**

```bash
pip install -r robot_policy_eval/requirements.txt
cd /path/to/your/checkout
python -m robot_policy_eval.run_evaluation --output-dir robot_policy_eval/outputs/run1
```

**Your data (trajectory JSON):** see [`paper/CROSS_MODEL_EVAL.md`](paper/CROSS_MODEL_EVAL.md). If you only have **SData** (no logged EE poses), build JSON from the manifest + movement YAML:

```bash
python -m robot_policy_eval.tools.build_sdata_trajectories \
  --manifest sdata_vla_benchmark/manifests/sdata_manifest.csv \
  --repo-root . \
  --output robot_policy_eval/data/sdata_trajectories.json
python -m robot_policy_eval.cross_model \
  --dataset robot_policy_eval/data/sdata_trajectories.json \
  --output-dir robot_policy_eval/outputs/sdata_eval
```

Outputs: `metrics_full.json`, `table_normal.csv`, `table_all_conditions.csv`, PNG figures (unless `--no-plots`).

## Programmatic

```python
from robot_policy_eval.config import ExperimentConfig
from robot_policy_eval.experiments.pipeline import run_default_experiment, save_outputs

suite, train_ds, test_ds = run_default_experiment(ExperimentConfig())
save_outputs(suite, Path("robot_policy_eval/outputs/run1"), ExperimentConfig())
```

Replace dummy policies with real `Policy` subclasses implementing `predict(observation_sequence) -> Trajectory`. For external VLAs that emit waypoints, tokens, or other formats, subclass [`ExternalTrajectoryPolicy`](policies/vla_adapter.py) and map outputs with `stack_to_trajectory` (see module docstring).

## Paper drop-ins

- **Comparison framing (read first):** [`paper/COMPARISON_FRAMEWORK.md`](paper/COMPARISON_FRAMEWORK.md) — four axes, trade-offs vs VLAs, disclaimer on invalid direct accuracy comparison.
- **Comparative table (generated, no hand-copied numbers):** [`paper/generate_tables.py`](paper/generate_tables.py) writes `TABLE.md`, `table_comparative_evaluation_generated.tex`, and `metrics_table.json` into your `--output-dir` when you run `run_evaluation` (or `python -m robot_policy_eval.paper.generate_tables metrics_full.json`).
- **Capability-axis table (same data, grouped by axis):** [`paper/generate_capability_table.py`](paper/generate_capability_table.py) writes `TABLE_CAPABILITY.md` and `metrics_capability.json` (also emitted on full eval runs).
- **Metric definitions:** [`paper/METRICS.md`](paper/METRICS.md)
- **Figure spec (A/B/C):** [`paper/FIGURES.md`](paper/FIGURES.md)

After a run, combined figures include `fig_paper_trajectory_gt_physio_vla.png` and `fig_paper_error_vs_timestep.png`.
