# Capability-based comparison (physio vs VLAs)

This evaluation package measures **capabilities in a shared trajectory space**, not raw score parity across unrelated training distributions.

## Why we do not compare “accuracy vs accuracy”

Vision–language–action (VLA) models are trained on large, heterogeneous corpora (e.g., multi-robot, multi-task data). A **task-specific physio controller** is trained (or tuned) for **continuous contact** on a narrow domain (e.g., massage). VLAs may emit **discrete tokens**, **waypoints**, or **continuous actions** in **different action spaces** than your stack.

**Headline claim:** We do **not** treat a discrete instruction-classification score on RGB clips as interchangeable with **closed-loop trajectory success** on the physio task. If you report such a number elsewhere, label it clearly as a **different task** (appendix / qualitative), not as a direct competitor to trajectory success here.

**Paper-ready sentence (verbatim-friendly):**

> We do not directly compare raw performance numbers across models due to differences in training distribution and action representation. Instead, we evaluate along four axes—task success, precision, generalization, and robustness—after mapping each policy into a **common end-effector trajectory and force representation** relative to the same observations and ground truth.

## Shared task layer

We compare policies only after they produce (or are converted to) the same abstraction:

- **Goal:** Execute a reference massage trajectory with safe contact (pressure/force within bounds).
- **Subtasks reflected in metrics:** trajectory following, force consistency, safety (threshold violations). Region localization is implicit if your dataset encodes it in GT trajectories.

**Normalization:** All policies under test implement `Policy.predict(...) -> Trajectory` ([`data/types.py`](../data/types.py)): positions, orientations (RPY), force, timestamps. See [`policies/vla_adapter.py`](../policies/vla_adapter.py) for how to wrap external VLAs.

## Four comparison axes

| Axis | What it captures | Typical expectation in a trade-off narrative |
|------|------------------|---------------------------------------------|
| **Task success** | Fraction of episodes meeting pose + force success criteria | Strong for a well-tuned in-domain controller |
| **Precision** | L2 / DTW trajectory error, jerk, force deviation | **Physio policy often strongest** (fine control) |
| **Generalization** | Held-out subjects (episode-level success aggregated by subject) | **VLAs may excel** on breadth if evaluated on diverse tasks elsewhere; here we test **unseen subjects** on the **same** task |
| **Robustness** | Success under input noise / occlusion (and future: lighting, motion, if added to the pipeline) | **Mixed** — report curves, not a single winner |

Metrics are defined precisely in [`METRICS.md`](METRICS.md). Tables are **generated** from `metrics_full.json` — do not hand-copy cells.

## Trade-off framing (2×2 style)

Modern VLA benchmarking emphasizes **trade-offs**, not a single leaderboard row. A useful narrative grid:

| Scenario | Typical expectation |
|----------|---------------------|
| In-domain physio (clean, train-like) | Strong precision + success for the specialized controller |
| Cross-domain or instruction-only probes | VLAs may show breadth **on their own benchmarks** — not commensurate with our trajectory metrics unless mapped into `Trajectory` |
| Noisy / perturbed observations | Compare **robustness curves** (same metric definitions for all policies) |
| Precision-critical contact | Specialized controller should lead on trajectory + force error **when evaluated fairly** |

**Do not** conclude universal superiority from one metric; **do** report where each approach is expected to shine.

## Hybrid system (recommended experiment)

A **high-level** VLA (or planner) plus **low-level** physio refinement aligns with planner + controller trends:

- Code: [`HybridPolicy`](../policies/hybrid.py) — replace dummy sub-policies with real models.
- Evaluate **hybrid vs** standalone physio and standalone VLA on the **same** four axes and figures ([`FIGURES.md`](FIGURES.md)).

## Out of scope for this package

- Full **LIBERO** / **Open-X** sim harnesses — use separate environments; feed resulting trajectories here if you map them into `Trajectory`.
- **Discrete-command-only** benchmarks without trajectory mapping — appendix + disclaimer, not the main table.

## Related files

- [`CROSS_MODEL_EVAL.md`](CROSS_MODEL_EVAL.md) — **how to run multiple models on your trajectory JSON** (same metrics, subject splits)
- [`METRICS.md`](METRICS.md) — definitions
- [`generate_tables.py`](generate_tables.py) — comparative numeric table from `run_evaluation_suite`
- [`generate_capability_table.py`](generate_capability_table.py) — axis-grouped table (same data, clearer framing)
