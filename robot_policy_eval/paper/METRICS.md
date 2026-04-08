# Minimal metric definitions (paper text)

**Framing (VLAs vs physio):** Read [`COMPARISON_FRAMEWORK.md`](COMPARISON_FRAMEWORK.md) first — we compare **capabilities in a shared trajectory space**, not raw accuracy parity across unrelated training setups or discrete-only benchmarks.

Use verbatim or shorten.

- **Task success:** Fraction of trials in which the policy completes the target massage trajectory within predefined tolerances on pose and force, with no safety violations for that episode.

- **Trajectory error:** Mean L2 distance between predicted and ground-truth end-effector positions after temporal alignment (e.g., common time resampling); optionally report DTW as a path-level dissimilarity.

- **Smoothness:** Mean magnitude of jerk (third time derivative of position), computed from discrete samples with respect to timestamps.

- **Force deviation:** Mean absolute error between predicted and reference force along the trajectory (after alignment).

- **Generalization (table row):** Mean over held-out **subjects** of that subject’s episode-level task success (under the `normal` condition), expressed as a percentage — see `generate_tables.py` (`_generalization_score_percent`).

- **Robustness — noise test (table row):** Mean task success rate (%) averaged across all **position-noise** robustness conditions (`pos_noise_*` in the suite), per policy.

- **Safety violations:** Count of timesteps (or episodes) where predicted contact force exceeds a fixed safe threshold.
