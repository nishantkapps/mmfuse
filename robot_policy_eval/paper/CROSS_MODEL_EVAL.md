# Evaluating different models on your dataset (trajectory protocol)

This document describes **how we test multiple models on the same data** in `robot_policy_eval`. It is **not** a “frozen CLIP score on video frames” benchmark. It follows a **shared task layer** and **capability metrics** ([`COMPARISON_FRAMEWORK.md`](COMPARISON_FRAMEWORK.md)).

## SData → `trajectories.json` (MMFuse training data)

SData releases **RGB/audio** and **class labels**, not logged robot end-effector poses. To still run `cross_model` on the same cohort, generate a **reference trajectory JSON** from your manifest + movement YAML (shared semantics with the movement head):

```bash
python -m robot_policy_eval.tools.build_sdata_trajectories \
  --manifest sdata_vla_benchmark/manifests/sdata_manifest.csv \
  --repo-root . \
  --output robot_policy_eval/data/sdata_trajectories.json
```

Optional: `--movement-config config/sdata_movement_config.yaml`, `--split test`, `--max-rows 100`.

Each CSV row becomes one episode; **subject_id** is parsed from the path (e.g. `p041`). Ground-truth paths are **synthesized** from `delta_along` / `delta_lateral` / `magnitude` — **not** motion capture. State this in the paper; use for **comparative** metrics once every policy maps into `Trajectory`.

## What you need

1. **Dataset:** A JSON file of trajectories compatible with [`data/json_io.py`](../data/json_io.py):
   - Either a top-level list of trajectory dicts, or `{"trajectories": [...]}`.
   - Each trajectory must include `subject_id` (and ideally `episode_id`) so we can **hold out subjects** for generalization.

2. **Models:** Each candidate is a [`Policy`](../policies/base.py): `predict(observation) -> Trajectory`, where `Trajectory` is end-effector pose + force over time ([`data/types.py`](../data/types.py)).

3. **Observation:** The evaluator passes `{"ground_truth": trajectory}` (and noisy/occluded variants). Your policy may ignore fields you do not use, but must return a full `Trajectory` for metrics.

## What gets measured (same for every model)

- **Task success** — thresholded pose + force safety ([`METRICS.md`](METRICS.md))
- **Precision** — L2, DTW, jerk, force error
- **Generalization** — test only on **held-out subjects**
- **Robustness** — input position noise and timestep occlusion sweeps

VLAs that natively output tokens or waypoints must be **mapped** into this representation ([`policies/vla_adapter.py`](../policies/vla_adapter.py)). State clearly if a mapping is a **proxy** (appendix-level claim).

## How to run

**1. Demo policies on your JSON (CLI)**

```bash
python -m robot_policy_eval.cross_model \
  --dataset /path/to/trajectories.json \
  --output-dir robot_policy_eval/outputs/my_run \
  --policies dummy_physio dummy_vla hybrid
```

Optional: `--test-fraction 0.3`, or `--test-subjects subj_a subj_b`, `--no-plots`, `--seed 42`.

**2. Your own policies (Python API)**

```python
from pathlib import Path
from robot_policy_eval.config import ExperimentConfig
from robot_policy_eval.cross_model import evaluate_policies_on_trajectory_dataset
from robot_policy_eval.experiments.pipeline import save_outputs

from my_models import PhysioPolicy, MyVLAWrappedAsTrajectory  # your implementations

policies = [PhysioPolicy(...), MyVLAWrappedAsTrajectory(...)]
suite, train_ds, test_ds = evaluate_policies_on_trajectory_dataset(
    policies,
    Path("data/my_trajectories.json"),
    ExperimentConfig(),
    test_fraction=0.33,
    seed=42,
)
save_outputs(suite, Path("robot_policy_eval/outputs/my_run"), ExperimentConfig())
```

Replace imports with your modules. Every policy is evaluated with **identical** loops and aggregation.

**3. MMFuse + registered baselines (same trajectory metrics)**

Install extras: `pip install -r robot_policy_eval/requirements-baselines.txt`.

```bash
python -m robot_policy_eval.sota \
  --dataset robot_policy_eval/data/sdata_trajectories.json \
  --repo-root . \
  --mmfuse-checkpoint /path/to/ckpt.pt \
  --baselines mmfuse clip_rt openclip_b32 hybrid_demo \
  --output-dir robot_policy_eval/outputs/sota
```

Optional: `openvla`, `openclip_l14`, `rt1_stub`, `saycan_stub`, etc. Keys and display names: [`baselines/factory.py`](../baselines/factory.py).

## Hybrid (planner + controller)

To show **VLA coarse plan + your low-level tracker**, instantiate [`HybridPolicy`](../policies/hybrid.py) with your high-level and refiner policies, add it to `policies`, and compare against each branch alone on the **same** test split.

## What not to claim

Do **not** equate these trajectory metrics with **instruction classification accuracy** on unrelated RGB benchmarks. If you also run such a benchmark, keep it in an appendix and cite the framing in [`COMPARISON_FRAMEWORK.md`](COMPARISON_FRAMEWORK.md).
