# Figure layout (required three panels)

## Primary differentiator: hybrid vs standalone

The strongest **experimental** story for “physio + VLA literature” is often **not** a single leaderboard cell, but a **hybrid system**: use a high-level model (VLA-style planner, language-conditioned policy, or coarse trajectory) to propose a **rough plan**, then refine with the **task-specific low-level controller** (`HybridPolicy` in code).

**What to show:** On the **same** held-out episodes and the **same** metrics as panels (A–C), report **hybrid vs** standalone physio and standalone VLA:

- Trajectory and error plots should include **three** or **four** curves: GT, physio, VLA, **hybrid** (see `plot_3d_three_trajectories(..., hybrid=...)` and `plot_error_vs_timestep_three_models` — extend labels if you add a fourth series in your fork).
- Main text: interpret as **complementarity** (planner breadth + executor precision), not as “VLA wins everywhere.”

Replace `DummyVLAPolicy` / `DummyPhysioPolicy` inside `HybridPolicy` with your real models when running experiments.

## (A) Trajectory comparison

- **Content:** Ground truth vs **your physio policy** vs **VLA** (and optionally hybrid as a fourth curve or separate figure).
- **Axes:** 3D $(x,y,z)$ or 2D projection (e.g., $x$–$y$ or arc-length vs height) — same units as calibration.
- **Style:** Distinct line styles / colors; mark start and goal; legend outside or caption identifies models.

**Implementation:** `robot_policy_eval.visualization.plots.plot_3d_three_trajectories(...)` saves a single PNG with GT + physio + VLA and optional **hybrid** overlay — use this for the hybrid experiment above.

## (B) Error over time

- **X-axis:** Timestep index or time (s).
- **Y-axis:** Position error (m), e.g., L2 distance between predicted and GT after per-step alignment / resampling.
- **Curves:** One line per model (physio, VLA, hybrid) on the **same** episode for direct comparison.

**Implementation:** `plot_error_time_three_models(...)`.

## (C) Robustness curve

- **X-axis:** Observation noise level $\sigma$ (or occlusion probability).
- **Y-axis:** Task success rate (\%) or binary success fraction aggregated over trials.
- **Curves:** One per policy; error bars optional (bootstrap over episodes).

**Implementation:** existing `plot_robustness_curve` from evaluation JSON; ensure all three policies appear.
