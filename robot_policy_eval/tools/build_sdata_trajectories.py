#!/usr/bin/env python3
"""
Build `trajectories.json` for `robot_policy_eval.cross_model` from an **SData CSV manifest**.

SData does **not** ship logged end-effector poses. This script creates a **reference
trajectory per row** by mapping the class label to `(delta_along, delta_lateral, magnitude)`
in `config/sdata_movement_config.yaml` (same semantics as MMFuse’s movement head) and
synthesizing a smooth EE path + force profile.

**This is a canonical proxy for evaluation**, not raw motion capture. Use it to run the
same capability metrics across policies; cite limitations in the paper (see
`paper/CROSS_MODEL_EVAL.md`).

Example:

  python -m robot_policy_eval.tools.build_sdata_trajectories \\
    --manifest sdata_vla_benchmark/manifests/sdata_manifest.csv \\
    --movement-config config/sdata_movement_config.yaml \\
    --repo-root . \\
    --output robot_policy_eval/data/sdata_trajectories.json
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import yaml

from robot_policy_eval.data.dataset import TrajectoryDataset
from robot_policy_eval.data.json_io import save_dataset_json
from robot_policy_eval.trajectory_synthesis import load_movements_yaml, trajectory_from_class_label


def _participant_from_cam_path(cam1: str) -> str:
    """e.g. .../part1/p041/p041_c1_part1.mp4 -> p041"""
    norm = cam1.replace("\\", "/")
    for seg in norm.split("/"):
        if re.match(r"^p\d+$", seg, re.I):
            return seg.lower()
    m = re.search(r"/(p\d+)/", norm, re.I)
    return m.group(1).lower() if m else "unknown"


def manifest_rows_to_trajectories(
    manifest_path: Path,
    movements_yaml: Path,
    *,
    repo_root: Path,
    split_filter: str | None = None,
    max_rows: int | None = None,
    n_timesteps: int = 48,
    dt: float = 0.02,
) -> TrajectoryDataset:
    repo_root = Path(repo_root).resolve()
    movements = load_movements_yaml(movements_yaml)

    trajectories: list[Trajectory] = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split_filter and row.get("split", "").strip() != split_filter:
                continue
            label = int(row["label"])
            sample_id = int(row["sample_id"])
            aug_v = int(row.get("aug_v", 0))
            cam1 = row["cam1"].strip()
            resolved = (repo_root / cam1).resolve()
            subj = _participant_from_cam_path(cam1)
            audio = row.get("audio", "").strip()
            tr = trajectory_from_class_label(
                label,
                movements,
                n_timesteps=n_timesteps,
                dt=dt,
                aug_v=aug_v,
                sample_id=sample_id,
                subject_id=subj,
                episode_id=f"sdata_{sample_id}",
                extra_meta={
                    "source": "sdata_movement_yaml_synthetic",
                    "sample_id": sample_id,
                    "split": row.get("split", ""),
                    "aug_v": aug_v,
                    "cam1": str(cam1),
                    "cam2": row.get("cam2", "").strip(),
                    "audio": audio,
                    "cam1_resolved": str(resolved),
                    "note": "GT trajectory synthesized from movement YAML; not logged robot data.",
                },
            )
            trajectories.append(tr)
            if max_rows is not None and len(trajectories) >= max_rows:
                break

    return TrajectoryDataset(trajectories, name="sdata_from_manifest")


def main() -> None:
    p = argparse.ArgumentParser(description="Export SData manifest → trajectory JSON for robot_policy_eval.")
    p.add_argument("--manifest", type=Path, required=True, help="CSV: sample_id,split,label,cam1,cam2,audio,aug_v")
    p.add_argument(
        "--movement-config",
        type=Path,
        default=None,
        help="YAML with movements[] (default: <repo-root>/config/sdata_movement_config.yaml)",
    )
    p.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Resolve relative paths in manifest")
    p.add_argument("--output", type=Path, required=True, help="Output JSON (TrajectoryDataset)")
    p.add_argument("--split", type=str, default=None, help="If set, only rows where split equals this (e.g. test)")
    p.add_argument("--max-rows", type=int, default=None, help="Debug: cap number of rows")
    p.add_argument("--n-timesteps", type=int, default=48)
    p.add_argument("--dt", type=float, default=0.02)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    movement_cfg = args.movement_config
    if movement_cfg is None:
        candidate = repo_root / "config" / "sdata_movement_config.yaml"
        if not candidate.is_file():
            print(
                f"Could not find default movement config at {candidate}. Pass --movement-config.",
                file=sys.stderr,
            )
            sys.exit(1)
        movement_cfg = candidate

    ds = manifest_rows_to_trajectories(
        args.manifest,
        movement_cfg,
        repo_root=repo_root,
        split_filter=args.split,
        max_rows=args.max_rows,
        n_timesteps=args.n_timesteps,
        dt=args.dt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_json(ds, args.output)
    print(f"Wrote {len(ds)} trajectories to {args.output.resolve()}")


if __name__ == "__main__":
    main()
