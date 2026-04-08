"""
Figures: 3D trajectories, error vs time, robustness curves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

from robot_policy_eval.data.types import Trajectory
from robot_policy_eval.metrics.trajectory_error import per_timestep_position_errors


def plot_3d_three_trajectories(
    gt: Trajectory,
    physio: Trajectory,
    vla: Trajectory,
    out_path: Path,
    hybrid: Trajectory | None = None,
    title: str = "GT vs Physio vs VLA",
    *,
    label_a: str = "Physio (ours)",
    label_b: str = "VLA",
    label_hybrid: str = "Hybrid",
) -> None:
    """
    Paper figure (A): single 3D plot — GT, physio, VLA [+ optional hybrid].
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    def _line(tr: Trajectory, style: str, label: str, lw: float = 1.8):
        p = tr.positions
        if len(p) == 0:
            return
        ax.plot(p[:, 0], p[:, 1], p[:, 2], style, label=label, linewidth=lw)

    g = gt.positions
    ax.plot(g[:, 0], g[:, 1], g[:, 2], "k-", label="GT", linewidth=2.4)
    _line(physio, "C0-", label_a)
    _line(vla, "C1--", label_b)
    if hybrid is not None:
        _line(hybrid, "C2-.", label_hybrid, lw=1.6)
    ax.scatter(g[0, 0], g[0, 1], g[0, 2], c="green", marker="o", s=45, zorder=5)
    ax.scatter(g[-1, 0], g[-1, 1], g[-1, 2], c="blue", marker="s", s=45, zorder=5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_error_vs_timestep_three_models(
    gt: Trajectory,
    preds: dict[str, Trajectory],
    out_path: Path,
    n_resample: int = 128,
) -> None:
    """
    Paper figure (B): trajectory error vs timestep index (0..N-1) for multiple models on one axes.
    `preds` keys are legend labels (e.g. 'Physio (ours)', 'VLA', 'Hybrid').
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    styles = ["C0-", "C1--", "C2-.", "C3:"]
    for i, (name, pred) in enumerate(preds.items()):
        _, err = per_timestep_position_errors(pred, gt, n_resample=n_resample)
        steps = np.arange(len(err))
        sty = styles[i % len(styles)]
        ax.plot(steps, err, sty, label=name, linewidth=1.3)
    ax.set_xlabel("Timestep (resampled)")
    ax.set_ylabel("Position L2 error (m)")
    ax.set_title("Trajectory error vs time")
    ax.grid(True, alpha=0.35)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_3d_trajectories(
    gt: Trajectory,
    pred: Trajectory,
    out_path: Path,
    title: str = "GT vs predicted EE path",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    g = gt.positions
    p = pred.positions
    ax.plot(g[:, 0], g[:, 1], g[:, 2], "g-", label="GT", linewidth=2)
    ax.plot(p[:, 0], p[:, 1], p[:, 2], "r--", label="Pred", linewidth=1.5)
    ax.scatter(g[0, 0], g[0, 1], g[0, 2], c="green", marker="o", s=40, label="start")
    ax.scatter(g[-1, 0], g[-1, 1], g[-1, 2], c="blue", marker="s", s=40, label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_vs_time(
    gt: Trajectory,
    pred: Trajectory,
    out_path: Path,
    n_resample: int = 128,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t, err = per_timestep_position_errors(pred, gt, n_resample=n_resample)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, err, "b-", linewidth=1.2)
    ax.fill_between(t, err, alpha=0.25)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position L2 error (m)")
    ax.set_title("Per-timestep error (after resampling)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_robustness_curve(
    suite_result: dict[str, Any],
    out_path: Path,
    noise_prefix: str = "pos_noise_",
) -> None:
    """
    X: noise sigma (parsed from aggregate keys), Y: success_rate per policy.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ag = suite_result.get("aggregates", {})
    noise_levels: list[float] = []
    for conds in ag.values():
        for k in conds:
            if k.startswith(noise_prefix):
                try:
                    noise_levels.append(float(k.replace(noise_prefix, "")))
                except ValueError:
                    pass
    noise_levels = sorted(set(noise_levels))

    fig, ax = plt.subplots(figsize=(8, 5))
    for pname, conds in ag.items():
        ys = []
        xs = []
        for nl in noise_levels:
            key = f"{noise_prefix}{nl}"
            if key in conds:
                xs.append(nl)
                ys.append(conds[key].get("success_rate", conds[key].get("success", 0.0)))
        if xs:
            ax.plot(xs, [100.0 * float(y) for y in ys], marker="o", label=pname)
    ax.set_xlabel("Input position noise σ (m)")
    ax.set_ylabel("Task success rate (%)")
    ax.set_title("Robustness: success vs observation noise")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_occlusion_curve(
    suite_result: dict[str, Any],
    out_path: Path,
    occ_prefix: str = "occlusion_",
) -> None:
    ag = suite_result.get("aggregates", {})
    occ_levels: list[float] = []
    for conds in ag.values():
        for k in conds:
            if k.startswith(occ_prefix):
                try:
                    occ_levels.append(float(k.replace(occ_prefix, "")))
                except ValueError:
                    pass
    occ_levels = sorted(set(occ_levels))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for pname, conds in ag.items():
        ys, xs = [], []
        for o in occ_levels:
            key = f"{occ_prefix}{o}"
            if key in conds:
                xs.append(o)
                ys.append(conds[key].get("success_rate", conds[key].get("success", 0.0)))
        if xs:
            ax.plot(xs, [100.0 * float(y) for y in ys], marker="s", label=pname)
    ax.set_xlabel("Random timestep drop probability")
    ax.set_ylabel("Task success rate (%)")
    ax.set_title("Robustness: occlusion")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
