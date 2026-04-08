"""Force channel comparison after temporal alignment."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from robot_policy_eval.data.types import Trajectory


def mean_absolute_force_error(pred: Trajectory, gt: Trajectory, n_resample: int = 64) -> float:
    """MAE between force scalars after resampling both trajectories to common time grids."""
    if len(pred) == 0 or len(gt) == 0:
        return 0.0

    def _resample_f(tr: Trajectory) -> tuple[np.ndarray, np.ndarray]:
        t = tr.timestamps
        f = tr.forces
        t_min, t_max = float(t[0]), float(t[-1])
        if t_max <= t_min:
            t_max = t_min + 1e-6
        t_new = np.linspace(t_min, t_max, n_resample)
        fi = interp1d(t, f, kind="linear", fill_value="extrapolate")
        return t_new, np.asarray(fi(t_new), dtype=np.float64)

    _, fp = _resample_f(pred)
    _, fg = _resample_f(gt)
    return float(np.mean(np.abs(fp - fg)))
